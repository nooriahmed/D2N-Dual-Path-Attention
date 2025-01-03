import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Function to plot training & validation loss and accuracy
def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history.get('accuracy')
    val_accuracy = history.history.get('val_accuracy')

    epochs = range(1, len(loss) + 1)

    # Plot training & validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    if accuracy and val_accuracy:
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        print("Accuracy metrics not available in history object.")

    plt.tight_layout()
    plt.show()

# Function to preprocess a single image from file path
def preprocess_image(image_path, target_size):
    """
    Preprocesses an image from a given file path to match the model's input size.

    Parameters:
    - image_path: Path to the input image
    - target_size: Size the model expects for input images (e.g., (224, 224))

    Returns:
    - Preprocessed image ready for model prediction
    """
    # Load the image and resize it
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)

    # Normalize the image
    image = image.astype('float32') / 255.0

    # Add batch dimension (1, height, width, channels)
    image = np.expand_dims(image, axis=0)

    return image

# Function to load a ground truth image from file path
def load_ground_truth(ground_truth_path, target_size):
    """
    Loads and preprocesses the ground truth image from the given file path.

    Parameters:
    - ground_truth_path: Path to the ground truth image
    - target_size: Size the ground truth image should be resized to

    Returns:
    - Preprocessed ground truth image (2D array)
    """
    ground_truth = load_img(ground_truth_path, target_size=target_size, color_mode='grayscale')
    ground_truth = img_to_array(ground_truth)

    # Normalize the ground truth (optional, depending on format)
    ground_truth = ground_truth.astype('float32') / 255.0

    return ground_truth

# General function to predict and visualize from image paths
def predict_and_visualize_from_paths(model, image_path, ground_truth_path, target_size=(224, 224)):
    """
    Predicts the segmentation map for the given image and ground truth paths,
    and visualizes the input image, ground truth, and prediction.

    Parameters:
    - model: Trained segmentation model
    - image_path: Path to the input image
    - ground_truth_path: Path to the ground truth image
    - target_size: The input size the model expects (default is (224, 224))
    """

    # Preprocess the image and ground truth
    image_preprocessed = preprocess_image(image_path, target_size)
    ground_truth = load_ground_truth(ground_truth_path, target_size)

    # Predict using the model
    prediction = model.predict(image_preprocessed)[0]  # Remove batch dimension after prediction

    # Visualize the input image, ground truth, and prediction
    plt.figure(figsize=(12, 6))

    # Load the original image for display (not preprocessed)
    original_image = load_img(image_path)

    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(original_image)

    plt.subplot(1, 3, 2)
    plt.title('Ground Truth')
    plt.imshow(ground_truth.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    plt.imshow(prediction.squeeze(), cmap='gray')

    plt.show()

# Example usage
image_path = "image_4205.jpg"  # Replace with your image path
ground_truth_path = "mask_4205.jpg"  # Replace with your ground truth path

predict_and_visualize_from_paths(model, image_path, ground_truth_path, target_size=(60, 60))
