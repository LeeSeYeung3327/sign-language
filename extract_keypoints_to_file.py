import os
import numpy as np
import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands

def extract_keypoints_from_video(video_path, use_movement_filter=False, movement_threshold=0.05):
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    previous_hand_positions = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            frame_keypoints = np.zeros((42, 2), dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[hand_idx].classification[0].label
                    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
                    wrist = landmarks[0]

                    update = True
                    if use_movement_filter:
                        if len(previous_hand_positions) > hand_idx:
                            prev_wrist = previous_hand_positions[hand_idx]
                            movement = np.linalg.norm(wrist - prev_wrist)
                            update = movement > movement_threshold

                    if update:
                        if label == 'Left':
                            frame_keypoints[0:21] = landmarks
                        else:
                            frame_keypoints[21:42] = landmarks

                    if len(previous_hand_positions) <= hand_idx:
                        previous_hand_positions.append(wrist)
                    else:
                        previous_hand_positions[hand_idx] = wrist

            keypoints.append(frame_keypoints)

    cap.release()
    return np.array(keypoints)

def normalize_and_pad(kp, sequence_length):
    if kp.shape[0] == 0:
        kp = np.zeros((sequence_length, 42, 2), dtype=np.float32)

    normalized = []
    for frame in kp:
        left_wrist = frame[0].copy()
        right_wrist = frame[21].copy()
        frame[:21] -= left_wrist
        frame[21:] -= right_wrist
        normalized.append(frame)
    normalized = np.array(normalized)

    if normalized.shape[0] < sequence_length:
        pad = np.repeat(normalized[-1][np.newaxis, :], sequence_length - normalized.shape[0], axis=0)
        normalized = np.concatenate([normalized, pad], axis=0)
    else:
        idxs = np.linspace(0, normalized.shape[0] - 1, sequence_length).astype(int)
        normalized = normalized[idxs]
    return normalized

def filter_static_hand(sequence, movement_threshold=0.01, margin=0.002):
    left_energy = np.sum(np.linalg.norm(sequence[:, 0:21, :], axis=-1))
    right_energy = np.sum(np.linalg.norm(sequence[:, 21:42, :], axis=-1))
    if left_energy < 1e-4 or right_energy < 1e-4:
        return sequence

    diff = np.diff(sequence, axis=0)
    left_diff = np.linalg.norm(diff[:, 0:21, :], axis=-1)
    right_diff = np.linalg.norm(diff[:, 21:42, :], axis=-1)
    left_mean = np.mean(left_diff)
    right_mean = np.mean(right_diff)

    if abs(left_mean - right_mean) < margin:
        return sequence
    else:
        if left_mean < right_mean:
            sequence[:, 0:21, :] = 0
        else:
            sequence[:, 21:42, :] = 0
    return sequence

def process_videos_to_csv(data_dir, output_dir, sequence_length=64, movement_threshold=0.05, use_movement_filter=False):
    # aggregated CSV 파일의 경로
    aggregated_path = os.path.join(output_dir, 'aggregated_dataset.csv')
    os.makedirs(output_dir, exist_ok=True)

    # 첫 실행 시에는 헤더부터 기록 (파일이 없으면 새로 생성)
    header = ['label'] + [f'kp_{i}' for i in range(sequence_length * 42 * 2)]
    with open(aggregated_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        print(f"[process] '{label}' 처리 중...")

        for video_name in os.listdir(label_dir):
            if not video_name.endswith('.mp4'):
                continue

            video_path = os.path.join(label_dir, video_name)
            print(f"   [비디오 처리] {video_path}")

            kp = extract_keypoints_from_video(video_path, use_movement_filter, movement_threshold)
            kp = normalize_and_pad(kp, sequence_length)
            kp = filter_static_hand(kp)
            flat_kp = kp.flatten()
            row = [label] + flat_kp.tolist()

            # 각 영상이 처리될 때마다 aggregated CSV 파일에 행을 append 모드로 저장
            with open(aggregated_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"   [저장 완료] {video_path} 데이터가 aggregated CSV에 추가됨")

    print(f"[완료] 모든 데이터가 {aggregated_path}에 저장됨")

if __name__ == "__main__":
    data_dir = "./dataset"       # 예: ./dataset/감사합니다, ./dataset/안녕하세요 등
    output_dir = "./csv_output"
    sequence_length = 64
    movement_threshold = 0.05
    use_movement_filter = False

    process_videos_to_csv(data_dir, output_dir, sequence_length, movement_threshold, use_movement_filter)
