
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to preprocess a single image from file path
def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to load ground truth
def load_ground_truth(ground_truth_path, target_size):
    ground_truth = load_img(ground_truth_path, target_size=target_size, color_mode='grayscale')
    ground_truth = img_to_array(ground_truth)
    ground_truth = ground_truth.astype('float32') / 255.0
    return ground_truth

# Function to overlay mask on input image with transparency
def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlays the predicted mask on the input image with transparency.
    The diseased area is highlighted in red.

    Parameters:
    - image: Input image (numpy array, HxWx3)
    - mask: Predicted mask (numpy array, HxW)
    - alpha: Transparency level for the overlay (0 to 1)

    Returns:
    - Overlayed image
    """
    overlay = image.copy()
    mask = (mask > 0.5).astype('uint8')  # Threshold the mask
    color = np.array([1, 0, 0])  # Red color for the overlay
    for c in range(3):  # Apply the red overlay with transparency
        overlay[..., c] = overlay[..., c] * (1 - alpha) + mask * color[c] * alpha
    return overlay

# Function to predict and visualize segmentation results
def predict_and_visualize(model, image_path, ground_truth_path=None, target_size=(60, 60)):
    """
    Predicts the segmentation map for the input image, overlays the predicted mask,
    and visualizes the results.

    Parameters:
    - model: Trained segmentation model
    - image_path: Path to the input image
    - ground_truth_path: Path to the ground truth image (optional)
    - target_size: Input size the model expects
    """

    # Preprocess input image
    image_preprocessed = preprocess_image(image_path, target_size)
    original_image = img_to_array(load_img(image_path, target_size=target_size)) / 255.0

    # Predict using the model
    prediction = model.predict(image_preprocessed)[0].squeeze()

    # Overlay mask on original image
    overlayed_image = overlay_mask_on_image(original_image, prediction, alpha=0.6)

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(original_image)

    if ground_truth_path:
        ground_truth = load_ground_truth(ground_truth_path, target_size)
        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        plt.imshow(ground_truth.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(overlayed_image)

    plt.show()

# Example usage
image_path = "image_1199.jpg"  # Replace with your image path
ground_truth_path = "mask_1199.jpg"  # Replace with your ground truth path
predict_and_visualize(model, image_path, ground_truth_path, target_size=(60, 60))
