# %% 1
import cv2
import numpy as np
import pytesseract

labels = ["plaka"]

colors = ["0,255,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1))

# %%3
model = cv2.dnn.readNetFromDarknet("plaka.cfg",
                                   "custom.weights")
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
img = cv2.imread("car2.jpg")

img_width = img.shape[1]
img_height = img.shape[0]
# %%2

layers = model.getLayerNames()
output_layer = [layers[i-1] for i in model.getUnconnectedOutLayers()]

img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
model.setInput(img_blob)

detection_layers = model.forward(output_layer)

start_x = 0
start_y = 0
end_x = 0
end_y = 0
box_color = None
label = None

# %%4
for detection_layer in detection_layers:
    for object_detection in detection_layer:

        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        if confidence > 0.20:
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))

            end_x = start_x + box_width
            end_y = start_y + box_height

            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]

            label = "{}: {:.2f}%".format(label, confidence * 100)
            print("predicted object {}".format(label))

cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 1)
cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

img_plaka = img[start_y:end_y, start_x:end_x]
img_plaka = cv2.resize(img_plaka,(711,230))

result = np.zeros(img_plaka.shape, dtype=np.uint8)
hsv = cv2.cvtColor(img_plaka, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,0])
upper = np.array([179,100,130])
mask = cv2.inRange(hsv, lower, upper)

# Perform morph close and merge for 3-channel ROI extraction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
extract = cv2.merge([close,close,close])

# Find contours, filter using contour area, and extract using Numpy slicing
cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = w * h
    if area < 10000 and area > 1000:
        cv2.rectangle(img_plaka, (x, y), (x + w, y + h), (36,255,12), 3)
        result[y:y+h, x:x+w] = extract[y:y+h, x:x+w]

# Invert image and throw into Pytesseract
invert = 255 - result
data = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')
print(data)

cv2.imshow("Detection Window", img_plaka)
cv2.imshow("Car", img)

cv2.waitKey()
cv2.destroyAllWindows()























