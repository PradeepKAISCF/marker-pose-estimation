import cv2
import numpy as np
import os
import time
import winsound  # ✅ Beep sound for Windows
import random

def detect_markers(image):
    """ Detects green markers and returns the image with drawn markers + coordinates """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 50, 50])  
    upper_green = np.array([80, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_coordinates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            marker_coordinates.append((cX, cY))

        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "centroid", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image, marker_coordinates

def augment_image(oimage):
    """ Apply strong augmentations: rotation, scaling, and translation """
    h, w = oimage.shape[:2]

    # ✅ Random Rotation (-90° to 90°)
    angle = random.uniform(-90, 90)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    image = cv2.warpAffine(oimage, M_rot, (w, h))

    # ✅ Random Scaling (90% to 110%)
    scale = random.uniform(0.9, 1.1)
    M_scale = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    image = cv2.warpAffine(oimage, M_scale, (w, h))

    # ✅ Random Translation (-30 to 30 pixels)
    tx, ty = random.randint(-30, 30), random.randint(-30, 30)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(oimage, M_trans, (w, h))

    return image

output_folder = 'dataset'
augmented_folder = 'dataset_augmented'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(augmented_folder, exist_ok=True)

cap = cv2.VideoCapture(1)
frame_count = 438
coordinates_file = open(os.path.join(output_folder, 'coordinates.txt'), 'w')

while True:
    ret, frame = cap.read(0)
    if not ret:
        break

    # ✅ Save original image
    image_filename = f'frame_{frame_count}.jpg'
    cv2.imwrite(os.path.join(output_folder, image_filename), frame)

    # ✅ Apply marker detection
    result, marker_coordinates = detect_markers(frame)
    coordinates_file.write(f'{image_filename}: {marker_coordinates}\n')

    # ✅ Generate Augmented Images
    for i in range(3):  # Generate 3 augmentations per image
        augmented_image = augment_image(frame)
        aug_filename = f'frame_{frame_count}_aug_{i}.jpg'
        cv2.imwrite(os.path.join(augmented_folder, aug_filename), augmented_image)

    frame_count += 1

    cv2.imshow('Marker Detection', np.hstack([frame, result]))

    # ✅ Beep sound after capturing an image
    winsound.Beep(1000, 300)  # (frequency=1000 Hz, duration=300 ms)

    time.sleep(1.25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
coordinates_file.close()
