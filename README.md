import cv2
import numpy as np
import subprocess
import time

# 1. 휴대폰 화면 캡처 (ADB를 사용한 스크린샷)
def capture_screen_via_adb():
    """
    ADB 명령어를 사용해 휴대폰 화면 캡처.
    adb 연결 필요.
    """
    adb_command = "adb exec-out screencap -p"
    process = subprocess.Popen(adb_command.split(), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    if output:
        # ADB에서 캡처한 이미지를 numpy 배열로 변환
        image = np.frombuffer(output, np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        raise ValueError("ADB를 통한 화면 캡처 실패")

# 2. 카드 탐지 (OpenCV 이미지 프로세싱)
def detect_cards(frame):
    """
    입력된 화면에서 카드 영역을 탐지하여 좌표를 반환.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 카드라고 판단할 만한 영역 추출
    cards = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    return cards

# 3. 카드 섞기 패턴 분석
def analyze_shuffle(initial_positions, current_positions):
    """
    카드의 초기 위치와 현재 위치를 비교하여 섞기 패턴 분석.
    """
    if not initial_positions or not current_positions:
        return "분석 불가: 카드 데이터 부족"

    shuffle_result = []
    for i, (init, curr) in enumerate(zip(initial_positions, current_positions)):
        if init != curr:
            shuffle_result.append(f"카드 {i+1}: 이동 (초기 {init} -> 현재 {curr})")
        else:
            shuffle_result.append(f"카드 {i+1}: 이동 없음")
    return shuffle_result

# 4. 메인 실행 루프
if __name__ == "__main__":
    try:
        print("ADB를 통해 휴대폰 화면 캡처를 시작합니다. 'q'를 눌러 종료.")
        initial_positions = None

        while True:
            # 화면 캡처
            frame = capture_screen_via_adb()
            cards = detect_cards(frame)

            # 카드 좌표 추출
            current_positions = [card[:2] for card in cards]  # (x, y)만 사용
            if initial_positions is None:
                initial_positions = current_positions  # 첫 번째 프레임의 위치 저장

            # 분석 결과 출력
            shuffle_analysis = analyze_shuffle(initial_positions, current_positions)
            for result in shuffle_analysis:
                print(result)

            # 화면에 카드 영역 표시
            for (x, y, w, h) in cards:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 화면 출력
            cv2.imshow("Card Shuffle Detection", frame)

            # 'q'를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.5)  # 처리 속도 조절

    except Exception as e:
        print("오류 발생:", e)
    finally:
        cv2.destroyAllWindows()
