### best.pt 파일을 사용하여 웹캠으로 실시간 객체 탐지  ###

import os
import cv2
from ultralytics import YOLO

#clear_output: 주피터 노트북 환경에서 화면에 글씨가 무한정 길어지지 않게 이전 글씨를 지워주는 기능
from IPython.display import clear_output

#머신러닝 라이브러리 충돌로 인한 프로그램 강제 종료를 막아주는 안전장치
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # best.pt 파일의 경로를 자동으로 찾아 추론용 모델로 불러옴. 이 모델은 "객체(예:마우스, 펜)을 구별할 줄 아는 상태
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "runs", "detect", "train", "weights", "best.pt")

    print(f"🎯 가중치 로드: {model_path}")
    model = YOLO(model_path)

    #웹캠 연
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 장치를 인식할 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()  #cap.read(): 카메라에서 찰나의 순간(1프레임)을 사진처럼 찰칵 찍어서 frame에 저장
        if not ret:
            break

        clear_output(wait=True)
        print("🎥 YOLOv8 실시간 탐지 작동 중... (종료: 영상 창 클릭 후 q)")

        # 실시간 추론 (verbose=False로 하단 텍스트 로그 완벽 차단)
        # 찍은 사진(frame)을 모델에 넣어 분석
        results = model(frame, stream=True, verbose=False)

        annotated_frame = frame
        for result in results:
            annotated_frame = result.plot()  #네모 박스(바운딩 박스)와 물체 이름(라벨) 구
            summary = result.verbose()
            names = result.names

            print(summary)  #화면 아래 텍스트 창에 "현재 화면에 마우스 1개, 펜 2개가 보입니다" 같은 요약 텍스트를 출력
            print(names)    #모델이 알고 있는 전체 사물 이름의 목록(딕셔너리)을 출력

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame) #1장의 사진을 화면

        # cv2.waitKey(1): 팝업창이 떠 있는 상태에서 키보드 입력을 0.001초 기다림
        # 'q'를 누르면, 화면 멈춤
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #카메라 끄기, 띄워졌던 영상 팝업 닫
    cap.release()
    cv2.destroyAllWindows()
    clear_output(wait=False)
    print("👋 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
