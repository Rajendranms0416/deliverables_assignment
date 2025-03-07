import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Parameters
image_size = (224, 224)

# Load generated images
image_paths = sorted(glob("generated_image_*.png"))
processed_images = []

for img_path in image_paths:
    img = cv2.imread(img_path)  # Load image
    img = cv2.resize(img, image_size)  # Resize to 224x224
    img = (img / 255.0).astype(np.float32)  # Normalize to [0,1]

    # Convert to grayscale
    img_uint8 = (img * 255).astype(np.uint8)  # Convert to uint8
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    # Save preprocessed image
    processed_path = img_path.replace(".png", "_processed.png")
    cv2.imwrite(processed_path, img_gray)
    processed_images.append(img_gray)

# Display preprocessed images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(processed_images[i], cmap='gray')
    ax.axis('off')
plt.show()

print("Image preprocessing completed.")