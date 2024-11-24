import random
import cv2
from ultralytics import YOLO
import pyautogui

from speech_text import speak

width, height = pyautogui.size()

frame_wid = 1200
frame_hyt = 720

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")


# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracked_objects = []
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.track(source=[frame], conf=0.55, save=False)

    # print(detect_params)
    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        current_object_ids = []
        for i in range(len(detect_params[0])):
            print(f"detected object {i}")
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            # Define a simple structure to represent an object
            detected_object = {
                "class_id": clsID,
                "confidence": conf,
                "id": i,
                "type": class_list[int(clsID)],
            }
            print(detected_object)
            current_object_ids.append(clsID)
            if not any(obj["class_id"] == clsID and obj["id"] == i for obj in tracked_objects):
                speak("detected " + detected_object["type"])
                tracked_objects.append(detected_object)

            if int(clsID) <= len(detection_colors):

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)],
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )
        # remove the object that is not detected in the current frame
        print(current_object_ids)
        for obj in tracked_objects[:]:
            if obj["class_id"] not in current_object_ids:
                speak("lost " + obj["type"])
                tracked_objects.remove(obj)
    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
