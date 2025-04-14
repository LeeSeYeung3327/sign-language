import os
import cv2
import numpy as np
import time

# ======= 촬영 관련 파라미터 설정 =======
VIDEO_OUTPUT_TEMPLATE = "{word}_{env}_video.mp4"  # 저장될 파일명 포맷
MIN_DURATION = 3          # 최소 녹화 시간 (초)
MAX_DURATION = 5          # 최대 녹화 시간 (초)
FPS = 20                  # 예상 프레임 레이트
MOTION_THRESHOLD = 500    # 두 프레임 간 차이가 임계값 이하일 때를 '동작 없음'으로 판단
NO_MOTION_REQUIRED = int(FPS * 1)  # 약 1초치 프레임의 변화가 없으면 동작 멈춤으로 간주

# ======= 웹캠 녹화 함수 =======
def record_video(output_filename):
    """
    웹캠으로부터 영상을 녹화합니다.
    최소 MIN_DURATION 이상 녹화 후, 연속된 NO_MOTION_REQUIRED 프레임에서 
    모션 변화가 MOTION_THRESHOLD 미만이면 녹화를 종료하거나, 
    MAX_DURATION에 도달하면 강제 종료합니다.
    녹화된 영상은 output_filename으로 저장됩니다.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return False

    # 실제 FPS 취득 (0이면 기본 FPS 사용)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0:
        actual_fps = FPS

    frames = []
    prev_gray = None
    consecutive_no_motion = 0
    start_time = time.time()

    print(f"녹화 시작: 최소 {MIN_DURATION}초 이상 녹화 후, 동작이 멈추면 녹화를 종료합니다. 'q'로 강제 종료 가능합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        frames.append(frame)

        # 모션 감지를 위해 회색조 변환 후 차이 계산
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < MOTION_THRESHOLD:
                consecutive_no_motion += 1
            else:
                consecutive_no_motion = 0
        prev_gray = gray.copy()

        # 실시간 화면 표시
        cv2.imshow("Webcam Recording", frame)

        elapsed_time = time.time() - start_time

        # 최소 녹화 시간 후 모션 멈춤 감지
        if elapsed_time >= MIN_DURATION and consecutive_no_motion >= NO_MOTION_REQUIRED:
            print("동작 멈춤 감지, 녹화 종료합니다.")
            break

        # 최대 녹화 시간 경과 시 강제 녹화 종료
        if elapsed_time >= MAX_DURATION:
            print("최대 녹화 시간 경과, 녹화를 종료합니다.")
            break

        # 'q' 키 입력 시 강제 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자에 의해 녹화 중단됨.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 녹화된 프레임 저장
    if len(frames) > 0:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, actual_fps, (width, height))
        for f in frames:
            out.write(f)
        out.release()
        print(f"녹화 완료, {output_filename} 파일에 저장되었습니다.")
        return True
    else:
        print("저장할 프레임이 없습니다.")
        return False

# ======= 자동화 메인 함수 =======
def main():
    # 23개의 단어와 8개의 환경 정의 (여기서는 예시 단어와 환경 이름)
    words = [
        "hello", "thankyou", "please", "sorry", "yes", "no", "goodbye", "welcome",
        "morning", "night", "happy", "sad", "angry", "excuseme", "help", "stop",
        "start", "wait", "more", "less", "big", "small", "love"
    ]
    environments = ["env1", "env2", "env3", "env4", "env5", "env6", "env7", "env8"]

    # 기본 데이터셋 폴더 생성
    base_dir = "dataset"
    os.makedirs(base_dir, exist_ok=True)

    # 각 단어 폴더 생성 및 환경별 녹화 진행
    for word in words:
        word_dir = os.path.join(base_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        print(f"\n--- 단어: {word} 녹화 시작 ---")
        for env in environments:
            prompt_msg = f"[{word} - {env}] 녹화 준비: 엔터 키를 눌러 3~5초 동안 녹화를 시작하세요..."
            input(prompt_msg)
            # 파일명 형식: dataset/word/word_env_video.mp4
            output_filename = os.path.join(word_dir, f"{word}_{env}_video.mp4")
            success = record_video(output_filename)
            if not success:
                print(f"녹화 실패: {output_filename}")
            else:
                print(f"[{word} - {env}] 녹화 완료.")
            # 녹화 사이 잠시 대기
            time.sleep(1)

    print("\n모든 녹화 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
