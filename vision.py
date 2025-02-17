import jetson.inference
import jetson.utils
import cv2
import numpy as np

# Load the YOLO object detection network
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Open camera stream (CSI camera)
camera = jetson.utils.videoSource("csi://0")  # For USB camera: "v4l2:///dev/video0"
display = jetson.utils.videoOutput("display://0")  # Use "my_video.mp4" to save output

while True:

    img = camera.Capture()

    detections = net.Detect(img)

    img_cv = jetson.utils.cudaToNumpy(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)

    for detect in detections:
        ID = detect.ClassID
        confidence = detect.Confidence
        left, top, right, bottom = int(detect.Left), int(detect.Top), int(detect.Right), int(detect.Bottom)

        cv2.rectangle(img_cv, (left, top), (right, bottom), (0, 255, 0), 2)

        label = f"{net.GetClassDesc(ID)}: {confidence:.2f}"
        cv2.putText(img_cv, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display result
    cv2.imshow("Object Detection", img_cv)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
