import cv2
import numpy as np
import torch
import mediapipe as mp
from torch import nn

# ----------------------------
# (추가) 정적 손 모양 검증 함수 (실제 환경에 맞게 구현 필요)
# ----------------------------
def check_static_hand_shape(kp_seq):
    """
    손가락 모양 등 정적인 특징을 기반으로 수어 동작에 맞는지 체크합니다.
    여기서는 예제로 항상 True를 반환합니다.
    """
    return True

# ----------------------------
# 전처리 및 스무딩 함수들
# ----------------------------
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
        idxs = np.linspace(0, normalized.shape[0]-1, sequence_length).astype(int)
        normalized = normalized[idxs]
    return normalized

def filter_static_hand(sequence, movement_threshold=0.01):
    diff = np.diff(sequence, axis=0)  # (T-1, 42, 2)
    left_diff = np.linalg.norm(diff[:, 0:21, :], axis=-1)
    right_diff = np.linalg.norm(diff[:, 21:42, :], axis=-1)
    left_mean = np.mean(left_diff)
    right_mean = np.mean(right_diff)
    
    if left_mean < movement_threshold and right_mean < movement_threshold:
        print(f"Static hand condition met (left: {left_mean:.4f}, right: {right_mean:.4f}).")
        print("Using unfiltered keypoints for prediction as fallback.")
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
    avg_movement = np.mean(movement)
    return avg_movement

def smooth_keypoints_ema(kp_seq, alpha=0.7):
    T, _, _ = kp_seq.shape
    smoothed = np.copy(kp_seq)
    for t in range(1, T):
        smoothed[t] = alpha * kp_seq[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

# ----------------------------
# 프레임 처리 함수 (process_frame)
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

# ----------------------------
# 키포인트 처리 및 예측 함수 (process_keypoints)
# ----------------------------
def process_keypoints(kp_seq_buffer, sequence_length, model, device,
                      confidence_threshold=0.8, ambiguous_margin=0.2):
    kp_seq = np.array(kp_seq_buffer)  # (T, 42, 2)
    kp_seq = normalize_and_pad(kp_seq, sequence_length)
    filtered_seq = filter_static_hand(kp_seq, movement_threshold=0.01)
    if filtered_seq is None:
        print("Static hand condition detected. Fallback: Using unfiltered keypoints for prediction.")
        filtered_seq = kp_seq
    smoothed_seq = smooth_keypoints_ema(filtered_seq, alpha=0.7)
    
    # 정적인 손 모양(수어 단어) 체크를 수행하여, 해당 기준에 부합하지 않으면 예측 거부
    if not check_static_hand_shape(smoothed_seq):
        print("Static hand shape check failed. Skipping prediction.")
        return None, None

    # **움직임 임계치 조정**: "팔 내리고 가리키는" 동작과 같이 움직임이 작을 수 있으므로 threshold를 0.002로 낮춤.
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
            print(f"Prediction rejected: Low confidence (confidence={top_prob_val:.2f}, threshold={confidence_threshold}).")
            return None, None
        diff_prob = (torch.topk(probs, 2).values[0][0] - torch.topk(probs, 2).values[0][1]).item()
        if diff_prob < ambiguous_margin:
            print(f"Prediction rejected: Ambiguous gesture (difference={diff_prob:.2f} < ambiguous_margin={ambiguous_margin}).")
            return None, None
        return label_map[pred_class], top_prob_val

# ----------------------------
# 모델 정의 및 초기화
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {
    0: '가슴',
    1: '감사합니다',
    2: '구토',
    3: '귀',
    4: '기침',
    5: '다리',
    6: '맞다',
    7: '머리',
    8: '목',
    9: '무릎',
    10: '발',
    11: '발목',
    12: '복부',
    13: '손목뼈',
    14: '아니',
    15: '아프다',
    16: '안녕하세요',
    17: '어깨',
    18: '어지럽다',
    19: '열',
    20: '팔',
    21: '팔꿈치',
    22: '허리'
}
num_classes = len(label_map)
cnn_out_dim = 256
lstm_hidden_size = 128

model = CNN_LSTMModel(cnn_out_dim, lstm_hidden_size, num_classes, dropout=0.5).to(device)
model_path = "best_model_cnn_lstm.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# 손 검출 및 영상 입력 설정
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
cap = cv2.VideoCapture(0)
sequence_length = 64  # 한 동작(시퀀스)을 구성할 프레임 수

# ----------------------------
# 메인 루프 및 예측 안정성 강화 (버퍼 즉시 초기화)
# ----------------------------
keypoints_buffer = []      # 각 프레임의 키포인트 저장
prediction_buffer = []     # 연속 예측 결과를 통한 안정성 확보 버퍼
no_hand_msg_printed = False
no_hand_count = 0
no_hand_threshold = 30     # 약 30 프레임 (약 1초) 이상 손 검출 실패 시 버퍼 리셋

last_left_landmarks = None
last_right_landmarks = None
consecutive_threshold = 3   # 연속 3회 동일 예측이면 최종 확정
last_stable_prediction = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # 각 프레임 처리: 손 검출 및 키포인트 추출
    keypoints, last_left_landmarks, last_right_landmarks, no_hand_msg_printed = process_frame(
        frame, results, last_left_landmarks, last_right_landmarks, no_hand_msg_printed
    )
    
    # (옵션) 키포인트 시각화
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    cv2.imshow("Webcam", frame)
    
    keypoints_buffer.append(keypoints)
    
    # 손 검출 실패가 연속되면 버퍼 초기화
    if not results.multi_hand_landmarks:
        no_hand_count += 1
    else:
        no_hand_count = 0
    if no_hand_count > no_hand_threshold:
        print("Extended period with no hand detected. Resetting keypoints buffer.")
        keypoints_buffer = []
        prediction_buffer = []
        no_hand_count = 0
    
    # 충분한 프레임이 모이면 예측 수행 및 버퍼 즉시 초기화
    if len(keypoints_buffer) == sequence_length:
        prediction, conf = process_keypoints(keypoints_buffer, sequence_length, model, device,
                                             confidence_threshold=0.8, ambiguous_margin=0.2)
        if prediction is not None:
            prediction_buffer.append(prediction)
            print(f"Buffered prediction: {prediction}")
            # 예측 안정성: 버퍼 내 예측 결과가 모두 동일할 때만 최종 예측 출력
            if len(prediction_buffer) >= consecutive_threshold:
                if all(p == prediction_buffer[0] for p in prediction_buffer):
                    final_pred = prediction_buffer[0]
                    if final_pred != last_stable_prediction:
                        last_stable_prediction = final_pred
                        cv2.putText(frame, final_pred, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, (0, 255, 0), 2)
                        cv2.imshow("Webcam", frame)
                        print(f"Final prediction: {final_pred}")
                else:
                    print("Prediction unstable. No final output.")
                prediction_buffer = []  # 예측 버퍼 즉시 초기화
        else:
            print("Prediction failed or rejected due to conditions.")
            prediction_buffer = []  # 예측 실패 시 버퍼 초기화
        keypoints_buffer = []  # 시퀀스 버퍼 초기화
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
