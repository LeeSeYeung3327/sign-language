import cv2
import numpy as np
import joblib
import tensorflow as tf

# 저장된 모델과 라벨 인코더 불러오기
model = tf.keras.models.load_model("r2plus1d_video_model.h5")
encoder = joblib.load("label_encoder.pkl")

def extract_frames_from_video(video_path, sequence_length=16, img_height=112, img_width=112, channels=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return None
    frame_indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)
    frames = np.zeros((sequence_length, img_height, img_width, channels), dtype=np.float32)
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if i > 0:
                frames[i] = frames[i-1]
            else:
                frames[i] = np.zeros((img_height, img_width, channels), dtype=np.uint8)
        else:
            frames[i] = cv2.resize(frame, (img_width, img_height))
    cap.release()
    return frames / 255.0

def predict_video(video_path):
    frames = extract_frames_from_video(video_path)
    if frames is None:
        print("❌ 영상 프레임 추출 실패")
        return
    input_data = np.expand_dims(frames, axis=0)  # 배치 차원 추가
    prediction = model.predict(input_data)[0]
    pred_index = np.argmax(prediction)
    pred_class = encoder.classes_[pred_index]
    confidence = prediction[pred_index]
    print(f"예측 결과: {pred_class} ({confidence:.2f})")

# 실시간 예측 테스트
video_path = "test_samples/sample.mp4"  # 예측할 영상 경로
predict_video(video_path)
