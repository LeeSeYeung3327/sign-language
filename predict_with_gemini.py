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

# ----------------------------
# Helper Functions: 손 키포인트 전처리 및 스무딩
# ----------------------------
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
        print(f"Static hand condition met (left: {left_mean:.4f}, right: {right_mean:.4f}). Using unfiltered keypoints.")
        return None
    if left_mean < right_mean:
        sequence[:, 0:21, :] = 0
        print(f"Filtered: Left hand removed (mean {left_mean:.4f} vs {right_mean:.4f}).")
    else:
        sequence[:, 21:42, :] = 0
        print(f"Filtered: Right hand removed (mean {right_mean:.4f} vs {left_mean:.4f}).")
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

# ----------------------------
# 프레임 처리: MediaPipe Hands 결과 기반 keypoints 추출
# ----------------------------
def process_frame(frame, results, last_left, last_right, no_hand_printed):
    frame_keypoints = np.zeros((42, 2), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
            if handedness == "Left":
                frame_keypoints[0:21] = landmarks
                last_left = landmarks.copy()
            elif handedness == "Right":
                frame_keypoints[21:42] = landmarks
                last_right = landmarks.copy()
        no_hand_printed = False
    else:
        if not no_hand_printed:
            print("No hand detected; using previous landmarks.")
            no_hand_printed = True
        if last_left is not None:
            frame_keypoints[0:21] = last_left
        if last_right is not None:
            frame_keypoints[21:42] = last_right
    return frame_keypoints, last_left, last_right, no_hand_printed

def process_keypoints(kp_seq_buffer, sequence_length, model, device,
                      confidence_threshold=0.8, ambiguous_margin=0.2):
    kp_seq = np.array(kp_seq_buffer)
    kp_seq = normalize_and_pad(kp_seq, sequence_length)
    filtered_seq = filter_static_hand(kp_seq, movement_threshold=0.01)
    if filtered_seq is None:
        print("Static hand condition detected. Using unfiltered keypoints.")
        filtered_seq = kp_seq
    smoothed_seq = smooth_keypoints_ema(filtered_seq, alpha=0.7)
    if not check_static_hand_shape(smoothed_seq):
        print("Static hand shape check failed. Skipping prediction.")
        return None, None
    avg_move = compute_movement(smoothed_seq)
    print(f"Avg Movement: {avg_move:.4f}")
    if avg_move < 0.002:
        print("Insufficient movement detected. Skipping prediction.")
        return None, None
    x_tensor = torch.tensor(smoothed_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_prob, pred = torch.max(probs, dim=1)
        top_prob_val = top_prob.item()
        pred_class = pred.item()
        if top_prob_val < confidence_threshold:
            print(f"Prediction rejected: Confidence {top_prob_val:.2f} below threshold.")
            return None, None
        diff_prob = (torch.topk(probs, 2).values[0][0] - torch.topk(probs, 2).values[0][1]).item()
        if diff_prob < ambiguous_margin:
            print(f"Prediction rejected: Ambiguous gesture (diff={diff_prob:.2f}).")
            return None, None
        return label_map[pred_class], top_prob_val

# ----------------------------
# CNN-LSTM 모델 정의
# ----------------------------
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

# ----------------------------
# Gemini API를 활용한 문장 생성 함수 (Gemini 2.0 방식)
# ----------------------------
def convert_to_sentence(words: str) -> str:
    if not words.strip():
        print("convert_to_sentence: 입력된 단어가 없습니다.")
        return ""

    prompt_text = (
    f"다음 단어들을 활용해서, 환자가 진료실에서 의사에게 자연스럽게 말하는 문장을 한 문장으로 만들어줘. "
    f"예를 들어 '열'은 '열이 나요', '머리 아프다 열 어지럽다'는 '머리가 아프고 열이 나면서 어지러워요'처럼 표현해줘: {words}"
)


    # ✅ Gemini 2.0에 맞는 payload 형식
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    print("Gemini API 호출 payload:", payload)

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10)
        print("Gemini API 응답 코드:", response.status_code)
        print("Gemini API 응답 내용:", response.text)

        if response.status_code == 200:
            data = response.json()
            # ✅ 응답 파싱 - Gemini 2.0에서 기본 응답 구조는 이렇게 생김
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                parts = candidate.get("content", {}).get("parts", [])
                if parts:
                    sentence = parts[0].get("text", "").strip()
                    print("Gemini API로부터 문장 생성 결과:", sentence)
                    return sentence
            print("Gemini API: 예상하는 결과 필드가 없습니다.")
            return ""
        else:
            print(f"Gemini API error {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
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

# ----------------------------
# 전역 변수 및 모델/API 설정
# ----------------------------
GEMINI_API_KEY = "AIzaSyAXlJLTGqPLt0euMQSCHkBbvfIfqUP36G0"  # 반드시 실제 Gemini API 키로 변경하세요
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
    print("모델 파일을 찾을 수 없습니다.")

# 전역 변수: 예측된 단어들을 순서대로 저장 (리스트)
predicted_words = []

# ----------------------------
# VideoCamera 클래스: OpenCV를 통한 웹캠 캡처 및 수어 예측 (별도 쓰레드)
# ----------------------------
class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.keypoints_buffer = []
        self.SEQUENCE_LENGTH = 64
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        global predicted_words
        last_left, last_right = None, None
        no_hand_printed = False
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.latest_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            keypoints, last_left, last_right, no_hand_printed = process_frame(
                frame, results, last_left, last_right, no_hand_printed)
            self.keypoints_buffer.append(keypoints)
            if len(self.keypoints_buffer) >= self.SEQUENCE_LENGTH:
                prediction, conf = process_keypoints(
                    self.keypoints_buffer, self.SEQUENCE_LENGTH, model, device,
                    confidence_threshold=0.8, ambiguous_margin=0.2)
                if prediction is not None:
                    predicted_words.append(prediction)
                    print(f"Prediction: {prediction} (conf: {conf:.2f})")
                else:
                    print("Prediction failed or rejected.")
                self.keypoints_buffer = []
            time.sleep(0.03)
        self.cap.release()

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False

# ----------------------------
# FastAPI 애플리케이션 및 엔드포인트 설정
# ----------------------------
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_camera = None

@app.on_event("startup")
def startup_event():
    global video_camera
    video_camera = VideoCamera()
    print("VideoCamera 시작됨.")

@app.on_event("shutdown")
def shutdown_event():
    global video_camera
    if video_camera is not None:
        video_camera.stop()
        print("VideoCamera 종료됨.")

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
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    return StreamingResponse(generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/keywords")
def get_keywords():
    words_str = " ".join(predicted_words)
    return {"keywords": words_str}

@app.get("/gen_sentence")
def gen_sentence(word: str = None):
    global predicted_words
    # 쿼리 파라미터가 있으면 그걸 사용, 없으면 누적된 예측 단어 사용
    if word and word.strip():
        words_str = word.strip()
    else:
        words_str = " ".join(predicted_words)
    print("문장 생성을 위한 입력:", words_str)
    
    if not words_str:
        return {"sentence": "입력된 단어가 없습니다."}
    
    # Gemini API를 사용해 문장 생성
    sentence = convert_to_sentence(words_str)
    processed_sentence = remove_adjacent_duplicates(sentence)
    
    if not processed_sentence:
        processed_sentence = "문장 생성에 실패했습니다."
    
    # 문장 생성 후 항상 predicted_words 초기화
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
