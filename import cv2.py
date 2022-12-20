import cv2
cap= cv2.VideoCapture(0)

while(1):
    berhasil, frame = cap.read()
    cv2.imshow('tracking', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
