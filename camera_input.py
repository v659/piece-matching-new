import cv2
import time

def capture_and_show(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    cv2.imshow("Captured Image", frame)
    cv2.waitKey(1)


while True:
    capture_and_show(0)
    time.sleep(5)


cv2.destroyAllWindows()