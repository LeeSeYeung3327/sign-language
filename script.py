import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import AUC

# ============================ 설정 ============================
DATASET_DIR = "dataset"  # 폴더 구조: dataset/단어명/영상파일.mp4
SEQUENCE_LENGTH = 16  # 각 영상의 프레임 수를 16으로 통일
IMG_HEIGHT, IMG_WIDTH = 112, 112  # 리사이즈 후 크기
CROP_HEIGHT, CROP_WIDTH = 112, 112  # 중앙 크롭 크기
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 50
CONFIDENCE_THRESHOLD = 0.8

# ============================ 영상에서 프레임 추출 및 인터폴레이션 ============================
def extract_frames_from_video(video_path, sequence_length):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 필요한 프레임 인덱스를 미리 계산
    frame_indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 영상에서 해당 프레임으로 이동
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            # 중앙 크롭
            h_start = (IMG_HEIGHT - CROP_HEIGHT) // 2
            w_start = (IMG_WIDTH - CROP_WIDTH) // 2
            cropped_frame = frame[h_start:h_start + CROP_HEIGHT, w_start:w_start + CROP_WIDTH]
            frames.append(cropped_frame)
        else:
            frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS)))  # 없을 경우 0으로 채우기

    cap.release()

    # 부족한 프레임 채우기
    while len(frames) < sequence_length:
        frames.append(frames[-1])  # 마지막 프레임 반복

    return np.array(frames)

# ============================ 데이터 로드 ============================
def load_data_from_videos(dataset_dir):
    sequences, labels = [], []
    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        for file in os.listdir(label_path):
            if file.endswith(".mp4"):
                video_path = os.path.join(label_path, file)
                frames = extract_frames_from_video(video_path, SEQUENCE_LENGTH)
                if frames is not None:
                    sequences.append(frames)
                    labels.append(label)
    return np.array(sequences), labels

# ============================ 데이터 증강 ============================
def augment_image(image):
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((IMG_WIDTH // 2, IMG_HEIGHT // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (IMG_WIDTH, IMG_HEIGHT))
    brightness = np.random.uniform(0.7, 1.3)
    augmented_image = np.clip(rotated * brightness, 0, 255).astype(np.uint8)  # 정수형으로 변환
    return augmented_image

def augment_data(X):
    augmented_data = []
    for seq in X:
        augmented_seq = []
        for frame in seq:
            augmented_frame = augment_image(frame)
            augmented_seq.append(augmented_frame)
        augmented_data.append(np.array(augmented_seq))
    return np.array(augmented_data)

# ============================ 모델 정의 ============================
def build_model(num_classes):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        BatchNormalization(), MaxPooling3D((1, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        BatchNormalization(), MaxPooling3D((1, 2, 2)),
        Conv3D(128, (3, 3, 3), activation='relu'),
        BatchNormalization(), MaxPooling3D((1, 2, 2)),
        Flatten(), Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', AUC()])
    return model

# ============================ 실시간 예측 (영상 기반) ============================
def predict_video(model, encoder, video_path):
    if not os.path.exists(video_path):
        print(f"[경고] 테스트 영상이 존재하지 않습니다: {video_path}")
        return

    frames = extract_frames_from_video(video_path, SEQUENCE_LENGTH)
    if frames is None or len(frames) == 0:
        print("영상에서 충분한 프레임을 추출하지 못했습니다.")
        return

    input_data = np.expand_dims(frames, axis=0).astype('float32') / 255.0

    prediction = model.predict(input_data)[0]
    pred_index = np.argmax(prediction)
    pred_class = encoder.classes_[pred_index]
    confidence = prediction[pred_index]

    if confidence >= CONFIDENCE_THRESHOLD:
        print(f"예측: {pred_class} ({confidence:.2f})")
    else:
        print(f"예측이 불확실합니다. ({confidence:.2f})")

# ============================ 메인 함수 ============================
def main():
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    # 데이터 로드 및 증강
    X, labels = load_data_from_videos(DATASET_DIR)
    X = augment_data(X)

    # 정규화
    X = X.astype('float32') / 255.0

    # 라벨 인코딩
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    joblib.dump(encoder, "label_encoder.pkl")

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 모델 학습
    model = build_model(num_classes=len(np.unique(y)))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # TensorBoard 설정
    tensorboard = TensorBoard(log_dir='./logs')
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[early_stop, tensorboard])

    model.save("r2plus1d_video_model.h5")

    # 예측 테스트
    test_video = "test_samples/hello.mp4"  # 테스트용 영상 경로
    model = load_model("r2plus1d_video_model.h5")
    encoder = joblib.load("label_encoder.pkl")
    predict_video(model, encoder, test_video)

if __name__ == "__main__":
    main()
