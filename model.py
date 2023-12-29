import cv2
import requests
import numpy as np
import json
import os

net = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')

#URL for model
url = os.environ.get("MODEL_URL", "http://127.0.0.1:7000/frame-collect")

# Define the list of object classes

classes = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


def detect_objects_cv2(frame):
    # Resize frame to have a maximum width of 600 pixels
    height, width = frame.shape[:2]
    new_width = min(600, width)
    ratio = new_width / width
    new_height = int(height * ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network to get detections
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes on the original frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            class_name = classes[class_id]
            box = detections[0, 0, i, 3:7] * np.array([new_width, new_height, new_width, new_height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = '{}: {:.2f}%'.format(class_name, confidence * 100)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def detect_objects_nanodet(frame):
    # Resize frame to have a maximum width of 600 pixels
    form_data = {
            'frame': json.dumps(frame.tolist()),
        }
    response = requests.post(url, data=form_data)
    resp_json = response.json()
    return np.array(resp_json['res'])