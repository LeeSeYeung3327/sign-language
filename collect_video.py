import os
import cv2
import numpy as np
import re

def collect_env_videos(label_name, max_envs=50, save_dir='./dataset', duration_sec=3, fps=30):
    label_dir = os.path.join(save_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)

    # 기존 env 번호 확인
    existing_files = [f for f in os.listdir(label_dir) if f.endswith('.mp4')]
    existing_env_nums = []
    pattern = re.compile(rf'{re.escape(label_name)}_env(\d+)_video\.mp4')
    for f in existing_files:
        match = pattern.match(f)
        if match:
            existing_env_nums.append(int(match.group(1)))
    start_idx = max(existing_env_nums) + 1 if existing_env_nums else 1

    if start_idx > max_envs:
        print(f"{label_name}의 모든 환경(1~{max_envs}) 비디오가 이미 존재합니다.")
        return True

    for env_num in range(start_idx, max_envs + 1):
        input(f"[{label_name}] 환경 {env_num} 녹화를 준비하려면 엔터키를 누르세요...")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return False

        # 첫 번째 미리보기 단계: 녹화 시작 전 미리보기 화면 보여주기
        print(f"{label_name} - 환경 {env_num} 미리보기 모드: 엔터키를 누르면 3초간 녹화 시작합니다.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패.")
                break
            cv2.putText(frame, f"{label_name} - ENV {env_num} [미리보기]", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imshow("미리보기", frame)
            key = cv2.waitKey(1)
            if key == 13:  # Enter key
                print(f"{label_name} - 환경 {env_num} 녹화를 시작합니다.")
                cv2.destroyWindow("미리보기")
                break
            elif key == 27:  # ESC key
                print("녹화 준비 중단됨.")
                cap.release()
                cv2.destroyAllWindows()
                return False

        # 녹화 시작
        filename = f"{label_name}_env{env_num}_video.mp4"
        filepath = os.path.join(label_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (640,480))
        
        frame_count = 0
        max_frames = int(duration_sec * fps)
        print(f"{label_name} - 환경 {env_num} 녹화 중 (총 {duration_sec}초)...")
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패.")
                break
            out.write(frame)
            frame_count += 1

            cv2.putText(frame, f"{label_name} - ENV {env_num} [녹화 중]", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imshow("녹화 중", frame)
            if cv2.waitKey(1) == 27:  # ESC key to interrupt
                print("녹화 중단됨.")
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                return False

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"{filepath} 저장 완료.\n")

    return True

if __name__ == '__main__':
    labels = [
        "안녕하세요", "감사합니다", "아니", "맞다", "아프다",
        "머리", "복부", "가슴", "목", "팔",
        "다리", "기침", "열", "무릎", "발",
        "어지럽다", "허리", "팔꿈치", "손목뼈", "발목",
        "귀", "구토", "어깨"
    ]

    for label in labels:
        print(f"\n=== [{label}] 수집 시작 ===\n")
        success = collect_env_videos(label_name=label, max_envs=50)
        if not success:
            print(f"\n수집이 중단되어 [{label}] 이후 항목은 건너뜁니다.")
            break
