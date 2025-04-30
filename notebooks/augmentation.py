import cv2
import numpy as np
import os
import random

def random_rotation(image):
    angle = random.uniform(-180, 180)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def random_scaling(image):
    scale = random.uniform(0.8, 1.2)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    return cv2.warpAffine(image, M, (w, h))

def random_translation(image):
    tx, ty = random.randint(40, 80), random.randint(20, 80)
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))
def random_translation1(image):
    tx, ty = random.randint(-80, -40), random.randint(-80, -20)
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))

def apply_all_augmentations(image, filename):
    augmented_images = []
    
    rotated_image = random_rotation(image)
    scaled_image = random_scaling(image)
    translated_image = random_translation(image)
    
    augmented_images.append((rotated_image, f"rot_{filename}"))
    augmented_images.append((scaled_image, f"scaled_{filename}"))
    augmented_images.append((translated_image, f"translated_{filename}"))
    
    # Apply all three augmentations in sequence
    combined_image = random_translation1(image)
    augmented_images.append((combined_image, f"transalted1_{filename}"))

    combined_image = random_translation1(random_translation(random_scaling(random_rotation(image))))
    augmented_images.append((combined_image, f"combined_{filename}"))
    
    return augmented_images

input_dir = "dataset/"
output_dir = "dataset1_augmented/"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Skipping {filename} (unable to read)")
            continue
        
        augmented_images = apply_all_augmentations(image, filename)
        
        for augmented_image, augmented_filename in augmented_images:
            output_path = os.path.join(output_dir, f"aug_{augmented_filename}")
            cv2.imwrite(output_path, augmented_image)
            print(f"✅ Saved: {output_path}")

print("🎉 Augmentation complete! All images are saved in", output_dir)
