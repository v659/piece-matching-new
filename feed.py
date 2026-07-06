import cv2

# Try OBS Virtual Camera first
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Optional: reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("iPhone Feed", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()