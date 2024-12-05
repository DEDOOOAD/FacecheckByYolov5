import cv2
import torch
from datetime import datetime
from ultralytics import YOLO

# YOLOv5 모델 로드 (verbose=False로 설정하여 불필요한 로그 출력 억제)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)  # verbose=False

# 웹캠 시작
cap = cv2.VideoCapture(0)

# 출석을 한 번만 기록하기 위한 변수
attendance_marked = False

# 출석 기록을 텍스트 파일에 저장하는 함수
def mark_attendance(name):
    with open('attendance.txt', 'a') as file:
        file.write(f'{name} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

# 출석 체크 시스템 실행
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5로 얼굴 감지 (여기서 한 번만 모델 호출)
    results = model(frame)

    # 감지된 얼굴을 찾고 표시
    for *box, conf, cls in results.xywh[0]:
        if conf > 0.5:  # 신뢰도 기준 설정
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 출석을 한 번만 기록
            if not attendance_marked:
                mark_attendance('Test User')
                attendance_marked = True

    # 실시간 웹캠 화면 출력
    cv2.imshow('Face Detection and Attendance System', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 출석 기록이 저장된 텍스트 파일 확인
print("출석 기록이 'attendance.txt'에 저장되었습니다.")