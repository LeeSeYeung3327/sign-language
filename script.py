import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.initializers import HeNormal
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ============================ 설정 ============================
DATASET_DIR = "dataset"            # 데이터셋 폴더 경로
SEQUENCE_LENGTH = 16               # 영상당 사용할 프레임 개수
IMG_HEIGHT, IMG_WIDTH = 112, 112   # 각 프레임의 크기
CHANNELS = 3                       # 컬러 채널 (RGB)
BATCH_SIZE = 8
EPOCHS = 50
CONFIDENCE_THRESHOLD = 0.8
INITIAL_LEARNING_RATE = 0.001

# ============================ GPU 설정 ============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # GPU 메모리 점진적 할당
        print(f"🟢 GPU 사용 가능: {gpus}")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")
else:
    print("🔴 GPU를 사용할 수 없습니다.")

print("TensorFlow 버전:", tf.__version__)

# ============================ 프레임 추출 함수 ============================
def extract_frames_from_video(video_path, sequence_length):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 5:
        print(f"[경고] 프레임 수 부족 ({total_frames}): {video_path}")
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)
    frames = np.zeros((sequence_length, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[오류] 프레임 추출 실패: {video_path} 프레임 {idx}")
            if i > 0:
                frames[i] = frames[i - 1]  # 이전 프레임으로 대체
            else:
                frames[i] = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
        else:
            frames[i] = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    cap.release()
    return frames / 255.0

# ============================ 병렬 데이터 로드 함수 ============================
def load_data_from_label(label_path, label):
    sequences = []
    labels = []
    for file in os.listdir(label_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(label_path, file)
            frames = extract_frames_from_video(video_path, SEQUENCE_LENGTH)
            if frames is not None and frames.shape == (SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS):
                sequences.append(frames)
                labels.append(label)
            else:
                print(f"[오류] 유효하지 않은 영상 데이터: {video_path}")
    return sequences, labels

def load_data_from_videos(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"❌ 데이터셋 경로가 존재하지 않습니다: {dataset_dir}")
        return np.array([]), []
    
    sequences, labels = [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for label in os.listdir(dataset_dir):
            label_path = os.path.join(dataset_dir, label)
            if os.path.isdir(label_path):
                futures.append(executor.submit(load_data_from_label, label_path, label))
        for future in concurrent.futures.as_completed(futures):
            data, label_data = future.result()
            sequences.extend(data)
            labels.extend(label_data)
    return np.array(sequences), labels

# ============================ 데이터 증강 함수 ============================
def augment_image(image):
    angle = np.random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_WIDTH // 2, IMG_HEIGHT // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (IMG_WIDTH, IMG_HEIGHT))
    brightness = np.random.uniform(0.9, 1.1)
    cropped = rotated[10:-10, 10:-10]  # 랜덤 크롭
    resized = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    flipped = cv2.flip(blurred, 1)
    return np.clip(flipped * brightness, 0, 255).astype(np.uint8)

def augment_data(X):
    return np.array([
        np.array([augment_image(frame) for frame in sequence])
        for sequence in X
    ]).astype('float32') / 255.0

# ============================ R(2+1)D 블록 ============================
def r2plus1d_block(input_shape, filters):
    model = Sequential()
    model.add(Conv3D(filters, kernel_size=(1, 3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv3D(filters, kernel_size=(3, 1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    return model

# ============================ 모델 정의 함수 ============================
def build_model(num_classes):
    model = Sequential()
    model.add(r2plus1d_block((SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS), 32))
    model.add(r2plus1d_block((None, None, None, 32), 64))
    model.add(r2plus1d_block((None, None, None, 64), 128))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', AUC()])
    return model

# ============================ 학습 결과 시각화 함수 ============================
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ============================ 메인 함수 ============================
def main():
    print("1️⃣ 데이터 로드 중...")
    X, labels = load_data_from_videos(DATASET_DIR)
    if len(X) == 0:
        print("❌ 유효한 영상이 없습니다.")
        return

    print(f"총 샘플 수: {len(X)}, 클래스 수: {len(set(labels))}")
    print("클래스 분포:", Counter(labels))

    print("2️⃣ 클래스 가중치 계산 중...")
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))
    print("계산된 클래스 가중치:", class_weights_dict)

    print("3️⃣ 데이터 증강 중...")
    X_aug = augment_data(X)
    X_final = np.concatenate([X, X_aug], axis=0)
    y_final = labels + labels

    print("4️⃣ 라벨 인코딩 중...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_final)
    joblib.dump(encoder, "label_encoder.pkl")

    print("5️⃣ 학습 데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.2, stratify=y_encoded)

    print("6️⃣ 모델 학습 중...")
    model = build_model(num_classes=len(encoder.classes_))
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    csv_logger = CSVLogger('training_log.csv', append=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=[early_stop, tensorboard, checkpoint, reduce_lr, csv_logger],
        verbose=2
    )

    model.save("r2plus1d_video_model.h5")
    print("✅ 모델 저장 완료")

    print("7️⃣ 학습 결과 시각화")
    plot_training_history(history)

    print("8️⃣ 테스트 데이터 평가")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 AUC: {test_auc:.4f}")

    print("9️⃣ 분류 보고서")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))

    # (옵션) 테스트 비디오로 예측 확인
    test_video_path = "test_samples/sample.mp4"
    if os.path.exists(test_video_path):
        frames = extract_frames_from_video(test_video_path, SEQUENCE_LENGTH)
        if frames is not None:
            input_data = np.expand_dims(frames, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            pred_index = np.argmax(prediction)
            pred_class = encoder.classes_[pred_index]
            confidence = prediction[pred_index]
            print(f"테스트 비디오 예측 결과: {pred_class} ({confidence:.2f})")
        else:
            print("테스트 비디오 프레임 추출 실패")
    else:
        print("테스트 비디오 파일이 존재하지 않습니다.")

if __name__ == "__main__":
    main()
