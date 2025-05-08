import cv2
import time
import threading
import numpy as np
import torch
import mediapipe as mp
from torch import nn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
from collections import deque, Counter

DEBUG_MODE = False  # 필요 시 True로 로그 확인

# ------------------------------
# 전처리 및 스무딩 함수들
# ------------------------------
def check_static_hand_shape(kp_seq):
    return True

def normalize_and_pad(kp_seq, sequence_length):
    if kp_seq.shape[0] == 0:
        kp_seq = np.zeros((sequence_length, 42, 2), dtype=np.float32)
    normalized = []
    for frame in kp_seq:
        left_wrist = frame[0]
        right_wrist = frame[21]
        frame_norm = frame.copy()
        frame_norm[:21] = frame_norm[:21] - left_wrist
        frame_norm[21:] = frame_norm[21:] - right_wrist
        normalized.append(frame_norm)
    normalized = np.array(normalized)
    if normalized.shape[0] < sequence_length:
        pad = np.repeat(normalized[-1][np.newaxis, :], sequence_length - normalized.shape[0], axis=0)
        normalized = np.concatenate([normalized, pad], axis=0)
    else:
        idxs = np.linspace(0, normalized.shape[0] - 1, sequence_length).astype(int)
        normalized = normalized[idxs]
    return normalized

def filter_static_hand(sequence, movement_threshold=0.01):
    diff = np.diff(sequence, axis=0)
    left_diff = np.linalg.norm(diff[:, 0:21, :], axis=-1)
    right_diff = np.linalg.norm(diff[:, 21:42, :], axis=-1)
    left_mean = np.mean(left_diff)
    right_mean = np.mean(right_diff)
    if left_mean < movement_threshold and right_mean < movement_threshold:
        if DEBUG_MODE:
            print(f"Static hand condition: L={left_mean:.4f}, R={right_mean:.4f}.")
        return None
    if left_mean < right_mean:
        sequence[:, 0:21, :] = 0
    else:
        sequence[:, 21:42, :] = 0
    return sequence

def compute_movement(kp_seq):
    diff = np.diff(kp_seq, axis=0)
    movement = np.linalg.norm(diff, axis=-1)
    return np.mean(movement)

def smooth_keypoints_ema(kp_seq, alpha=0.7):
    T, _, _ = kp_seq.shape
    smoothed = np.copy(kp_seq)
    for t in range(1, T):
        smoothed[t] = alpha * kp_seq[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed

# ------------------------------
# 프레임 처리: 양손 landmark 추출 및 결합
# ------------------------------
def process_frame(frame, results, last_left, last_right, no_hand_printed):
    """
    두 손의 landmark를 추출합니다.
    - 왼손 landmark는 인덱스 0~20, 오른손은 21~41에 저장.
    - 만약 landmark 개수가 21개 미만이면 이전 값(last_left/last_right) 혹은 기본값으로 채웁니다.
    - 한쪽만 검출된 경우, 그 landmark를 복제하여 양손 데이터를 완성합니다.
    """
    combined_keypoints = np.zeros((42, 2), dtype=np.float32)
    left_points = None
    right_points = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].classification[0].label
            if DEBUG_MODE:
                print(f"Detected hand {i}: {label}")
            landmarks = results.multi_hand_landmarks[i].landmark
            if len(landmarks) < 21:
                if label == "Left" and last_left is not None:
                    points = last_left
                elif label == "Right" and last_right is not None:
                    points = last_right
                else:
                    points = np.zeros((21, 2), dtype=np.float32)
            else:
                points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
                points = np.clip(points, 0.0, 1.0)
            if label == "Left":
                left_points = points.copy()
            elif label == "Right":
                right_points = points.copy()
        no_hand_printed = False
    else:
        if not no_hand_printed and DEBUG_MODE:
            print("No hand detected; reusing previous landmarks.")
    
    if left_points is None and right_points is not None:
        left_points = right_points.copy()
    if right_points is None and left_points is not None:
        right_points = left_points.copy()

    if left_points is None:
        left_points = last_left if last_left is not None else np.zeros((21, 2), dtype=np.float32)
    if right_points is None:
        right_points = last_right if last_right is not None else np.zeros((21, 2), dtype=np.float32)

    combined_keypoints[0:21] = left_points
    combined_keypoints[21:42] = right_points

    last_left = left_points
    last_right = right_points

    return combined_keypoints, last_left, last_right, no_hand_printed

def process_keypoints(kp_seq_buffer, sequence_length, model_to_use, device,
                      confidence_threshold=0.75, ambiguous_margin=0.2, end_of_gesture=False):
    kp_seq = np.array(kp_seq_buffer)
    kp_seq = normalize_and_pad(kp_seq, sequence_length)
    filtered_seq = filter_static_hand(kp_seq, movement_threshold=0.01)
    if filtered_seq is None:
        filtered_seq = kp_seq
    smoothed_seq = smooth_keypoints_ema(filtered_seq, alpha=0.7)

    if not check_static_hand_shape(smoothed_seq):
        return None, None

    avg_move = compute_movement(smoothed_seq)
    if DEBUG_MODE:
        print(f"Avg Movement: {avg_move:.4f}")
    if not end_of_gesture and avg_move < 0.02:
        if DEBUG_MODE:
            print("Insufficient movement, skipping prediction.")
        return None, None

    x_tensor = torch.tensor(smoothed_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        output = model_to_use(x_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_prob, pred = torch.max(probs, dim=1)
        top_prob_val = top_prob.item()
        pred_class = pred.item()
        if DEBUG_MODE:
            print(f"Softmax Top Prob: {top_prob_val:.2f}")
        if top_prob_val < confidence_threshold:
            if DEBUG_MODE:
                print("Low confidence, skipping prediction.")
            return None, None
        diff_prob = (torch.topk(probs, 2).values[0][0] - 
                     torch.topk(probs, 2).values[0][1]).item()
        if DEBUG_MODE:
            print(f"Diff Prob: {diff_prob:.2f}")
        if diff_prob < ambiguous_margin:
            if DEBUG_MODE:
                print("Ambiguous gesture, skipping prediction.")
            return None, None
        return label_map[pred_class], top_prob_val

# ------------------------------
# CNN-LSTM 모델 정의
# ------------------------------
class CNN_LSTMModel(nn.Module):
    def __init__(self, cnn_out_dim, lstm_hidden_size, num_classes, dropout=0.5):
        super(CNN_LSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(32 * 42 * 2, cnn_out_dim),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=lstm_hidden_size,
                            num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(B * T, -1)
        cnn_features = self.cnn_fc(cnn_out)
        cnn_features = cnn_features.view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_features)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ------------------------------
# Gemini API 문장 생성 함수
# ------------------------------
def convert_to_sentence(words: str) -> str:
    if not words.strip():
        if DEBUG_MODE:
            print("No word input provided.")
        return ""
    prompt_text = (f"다음에 나열된 단어들만 사용해 자연스러운 문장을 만들어줘. "
                   f"주어진 단어들의 순서를 재배열하여 문장을 구성해줘: {words}")
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json"}
    if DEBUG_MODE:
        print("Gemini API payload:", payload)
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10)
        if DEBUG_MODE:
            print("Gemini API 응답 코드:", response.status_code)
            print("Gemini API 응답 내용:", response.text)
        if response.status_code == 200:
            data = response.json()
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                parts = candidate.get("content", {}).get("parts", [])
                if parts:
                    sentence = parts[0].get("text", "").strip()
                    if DEBUG_MODE:
                        print("Gemini API 결과:", sentence)
                    return sentence
            if DEBUG_MODE:
                print("Gemini API: No expected result.")
            return ""
        else:
            if DEBUG_MODE:
                print(f"Gemini API error {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        if DEBUG_MODE:
            print(f"Gemini API 호출 오류: {e}")
        return ""

def remove_adjacent_duplicates(sentence: str) -> str:
    words = sentence.split()
    if not words:
        return sentence
    result = [words[0]]
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    return " ".join(result)

# ------------------------------
# 글로벌 변수 및 모델/API 설정
# ------------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # 실제 API 키로 변경하세요.
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
label_map = {
    0: '가슴', 1: '감사합니다', 2: '구토', 3: '귀', 4: '기침',
    5: '다리', 6: '맞다', 7: '머리', 8: '목', 9: '무릎',
    10: '발', 11: '발목', 12: '복부', 13: '손목뼈', 14: '아니',
    15: '아프다', 16: '안녕하세요', 17: '어깨', 18: '어지럽다',
    19: '열', 20: '팔', 21: '팔꿈치', 22: '허리'
}
num_classes = len(label_map)
cnn_out_dim = 256
lstm_hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model_cnn_lstm.pth"
model = CNN_LSTMModel(cnn_out_dim, lstm_hidden_size, num_classes, dropout=0.5).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    if DEBUG_MODE:
        print("모델 파일이 존재하지 않습니다.")

example_input = torch.randn(1, 64, 1, 42, 2).to(device)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.eval()
predicted_words = []

# ------------------------------
# VideoCamera 클래스: 영상 캡처, 추론 및 제스처 상태 관리
# ------------------------------
class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            if DEBUG_MODE:
                print("웹캠 오픈 실패")
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 두 손 모두 감지
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
        self.keypoints_buffer = deque()
        self.SEQUENCE_LENGTH = 64
        self.no_hand_count = 0
        self.no_hand_threshold = 30  # 약 1초
        self.last_left = None
        self.last_right = None
        self.no_hand_printed = False
        self.hand_detected = False

        self.gesture_active = False
        self.gesture_start_time = None
        self.stable_gesture = False

        self.display_keypoints = None
        self.predicted_buffer = []
        self.consecutive_threshold = 3

        threading.Thread(target=self.update, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self.latest_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                self.hand_detected = True
                if self.gesture_start_time is None:
                    self.gesture_start_time = time.time()
                elif time.time() - self.gesture_start_time >= 1.0:
                    self.stable_gesture = True
                self.gesture_active = True
            else:
                self.hand_detected = False

            keypoints, self.last_left, self.last_right, self.no_hand_printed = process_frame(
                frame, results, self.last_left, self.last_right, self.no_hand_printed
            )
            with self.lock:
                self.keypoints_buffer.append(keypoints)

            if results.multi_hand_landmarks:
                current_keypoints, _, _, _ = process_frame(
                    frame, results, self.last_left, self.last_right, self.no_hand_printed
                )
                if self.display_keypoints is None:
                    self.display_keypoints = current_keypoints.copy()
                else:
                    alpha_display = 0.8
                    diff = np.linalg.norm(current_keypoints - self.display_keypoints, axis=1)
                    threshold = 0.5
                    new_coords = np.where(diff.reshape(-1, 1) < threshold, current_keypoints, self.display_keypoints)
                    self.display_keypoints = alpha_display * new_coords + (1 - alpha_display) * self.display_keypoints
                h, w, _ = frame.shape
                # 파란색 텍스트가 아닌, 원만 그리도록 수정함.
                for idx, (x, y) in enumerate(self.display_keypoints):
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    # cv2.putText 호출 제거: 좌표 텍스트를 출력하지 않음.
                with self.lock:
                    self.latest_frame = frame.copy()

            if not results.multi_hand_landmarks:
                self.no_hand_count += 1
            else:
                self.no_hand_count = 0
            if self.no_hand_count > self.no_hand_threshold:
                with self.lock:
                    self.keypoints_buffer.clear()
                self.no_hand_count = 0

            if results.multi_hand_landmarks and self.gesture_start_time is None:
                self.gesture_start_time = time.time()
            time.sleep(0.005)
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

    def inference_worker(self):
        global predicted_words
        while self.running:
            with self.lock:
                if not self.hand_detected and len(self.keypoints_buffer) > 0:
                    chunk = list(self.keypoints_buffer)
                    self.keypoints_buffer.clear()
                else:
                    chunk = None
            if chunk is not None:
                end_flag = not self.hand_detected
                prediction, conf = process_keypoints(
                    chunk, self.SEQUENCE_LENGTH, scripted_model, device,
                    confidence_threshold=0.8, ambiguous_margin=0.2, end_of_gesture=end_flag
                )
                if prediction is not None:
                    if self.gesture_active and self.stable_gesture:
                        predicted_words.append(prediction)
                        if DEBUG_MODE:
                            print(f"Final Prediction (gesture end): {prediction}")
                        self.predicted_buffer.clear()
                        self.gesture_active = False
                        self.gesture_start_time = None
                        self.stable_gesture = False
            else:
                time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return None

    def stop(self):
        self.running = False

# ------------------------------
# FastAPI 엔드포인트 정의
# ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
video_camera = None

@app.on_event("startup")
def startup_event():
    global video_camera
    video_camera = VideoCamera()
    if DEBUG_MODE:
        print("VideoCamera started.")

@app.on_event("shutdown")
def shutdown_event():
    global video_camera
    if video_camera is not None:
        video_camera.stop()
        if DEBUG_MODE:
            print("VideoCamera stopped.")

@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = video_camera.get_frame()
            if frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.005)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/keywords")
def get_keywords():
    words_str = " ".join(predicted_words)
    return {"keywords": words_str}

@app.get("/gen_sentence")
def gen_sentence(word: str = None):
    global predicted_words
    if word and word.strip():
        words_str = word.strip()
    else:
        words_str = " ".join(predicted_words)
    if DEBUG_MODE:
        print("Input for sentence generation:", words_str)
    if not words_str:
        return {"sentence": "입력된 단어가 없습니다."}
    sentence = convert_to_sentence(words_str)
    processed_sentence = remove_adjacent_duplicates(sentence)
    if not processed_sentence:
        processed_sentence = "문장 생성 실패."
    predicted_words.clear()
    return {"sentence": processed_sentence}

@app.get("/")
def read_index():
    index_path = os.path.join("templates", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html 파일을 찾을 수 없습니다."}

if __name__ == "__main__":
    uvicorn.run("predict_with_gemini:app", host="127.0.0.1", port=8000, reload=True)
