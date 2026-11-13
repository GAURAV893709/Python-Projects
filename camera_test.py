import cv2

print("Starting webcam test...")

camera = cv2.VideoCapture(0)

print("Camera opened:", camera.isOpened())

if camera.isOpened():
    print("Webcam connected successfully!")

    ret, frame = camera.read()

    print("Frame captured:", ret)

    if ret:
        cv2.imshow("Webcam Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Webcam not detected!")

camera.release()
print("Test completed.")