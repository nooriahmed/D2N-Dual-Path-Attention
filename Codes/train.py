
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, PReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import PReLU

# Paths for dataset
train_images_path = "path/trainx/" 
train_masks_path = "path/trainy/"
test_images_path = "path/testx/"  
test_masks_path = "path/testy/"    
val_images_path = "path/validationx/" 
val_masks_path = "path/validationy/"  

# Parameters
IMAGE_SIZE = (60, 60)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
DROPOUT_RATE = 0.3

# Function to preprocess images and masks
def preprocess_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_jpeg(image, channels=3)
    mask = tf.image.decode_jpeg(mask, channels=1)

    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    mask = tf.image.resize(mask, IMAGE_SIZE) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)
    return image, mask

# Function to create tf.data dataset
def create_dataset(image_dir, mask_dir, domain_label):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    # Create a dataset of image paths and mask paths
    path_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Map the preprocessing function. This now returns (image, mask)
    processed_dataset = path_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    # Create a dataset of dummy domain labels
    domain_label_dataset = tf.data.Dataset.from_tensor_slices(tf.fill(len(image_paths), domain_label))

    # Zip the processed image/mask dataset with the domain label dataset
    # This will yield (image, mask, domain_label) initially
    dataset = tf.data.Dataset.zip((processed_dataset.map(lambda img, mask: (img, mask)), domain_label_dataset))

    # Reformat the dataset to yield (inputs, targets) where targets is a dictionary
    # This matches the expected input for model.fit with multiple outputs
    dataset = dataset.map(lambda img_mask, domain_label: (img_mask[0], {'segmentation_output': img_mask[1], 'domain_output': domain_label}))

    # Apply batching and prefetching
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    return dataset

# Create datasets with domain labels
# Assuming training data is from domain 0 and validation/test data is from domain 1
train_dataset = create_dataset(train_images_path, train_masks_path, domain_label=0)
test_dataset = create_dataset(test_images_path, test_masks_path, domain_label=1)
val_dataset = create_dataset(val_images_path, val_masks_path, domain_label=1)

# Function to align dataset sizes
def align_dataset_sizes(train_ds, val_ds, test_ds):
    train_count = tf.data.experimental.cardinality(train_ds).numpy()
    # Take the same number of batches from validation and test datasets
    val_ds = val_ds.take(train_count)
    test_ds = test_ds.take(train_count)
    return train_ds, val_ds, test_ds

# Align dataset sizes
train_dataset, val_dataset, test_dataset = align_dataset_sizes(train_dataset, val_dataset, test_dataset)

# Gradient Reversal Mechanism (GRM)
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def call(self, inputs, **kwargs):
        @tf.custom_gradient
        def grad_reverse(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return grad_reverse(inputs)

# Cross-domain attention module (CDAM)
def cross_domain_attention(global_features, local_features):
    combined = Concatenate()([global_features, local_features])
    attention = Conv2D(1, (1, 1), activation='sigmoid')(combined)
    global_enhanced = Multiply()([global_features, attention])
    local_enhanced = Multiply()([local_features, attention])
    return global_enhanced, local_enhanced

# Domain Classifier
def domain_classifier(feature_map):
    x = SeparableConv2D(64, (3, 3), padding='same')(feature_map)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SeparableConv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    # Add Global Average Pooling to reduce feature map to a single value per sample
    x = GlobalAveragePooling2D()(x)
    return x

# Dual-path U-Net model with Domain Adaptation
def dual_path_unet_with_da(input_shape=(60, 60, 3), dropout_rate=0.5, lambda_=1.0):
    inputs = Input(input_shape)

    # Global Path
    g1 = SeparableConv2D(64, (3, 3), padding='same')(inputs)
    g1 = BatchNormalization()(g1)
    g1 = PReLU()(g1)
    g1 = MaxPooling2D((2, 2))(g1)

    g2 = SeparableConv2D(128, (3, 3), padding='same')(g1)
    g2 = BatchNormalization()(g2)
    g2 = PReLU()(g2)
    g2 = MaxPooling2D((2, 2))(g2)

    # Local Path
    l1 = SeparableConv2D(64, (3, 3), padding='same')(inputs)
    l1 = BatchNormalization()(l1)
    l1 = PReLU()(l1)
    l1 = MaxPooling2D((2, 2))(l1)

    l2 = SeparableConv2D(128, (3, 3), padding='same')(l1)
    l2 = BatchNormalization()(l2)
    l2 = PReLU()(l2)
    l2 = MaxPooling2D((2, 2))(l2)

    # CDAM
    g2, l2 = cross_domain_attention(g2, l2)
    merged = Concatenate()([g2, l2])

    # Bottleneck with residual
    shortcut = merged
    x = SeparableConv2D(256, (3, 3), padding='same')(merged)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = PReLU()(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, g1])
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, inputs])
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    # Segmentation output
    segmentation_outputs = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(x)

    # Domain adaptation path
    domain_grl = GradientReversalLayer(lambda_)(x)
    # The domain_classifier function already includes GlobalAveragePooling2D
    domain_output = domain_classifier(domain_grl)
    # Add a final Dense layer for the binary classification output (domain 0 or 1)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(domain_output)


    model = Model(inputs=inputs, outputs=[segmentation_outputs, domain_output])
    return model


optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

# Compile model
model = dual_path_unet_with_da()
model.compile(
    optimizer=optimizer,
    loss={
        'segmentation_output': 'binary_crossentropy',
        'domain_output': keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    },
    loss_weights={
        'segmentation_output': 1.0,
        'domain_output': 0.1  # Smaller weight for domain loss
    },
    metrics={
        'segmentation_output': ['accuracy'],
        'domain_output': ['accuracy']
    }
)


# Model summary
model.summary()


# Data Augmentation (optional, keep as is)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Fit the model
# The dataset now yields (image, (mask, domain_label)), matching the model's output structure
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=1)

# Save weights
model.save_weights('dual_path_unet_weights-brain.weights.h5') # Changed the filename to include '.weights.h5'
print("Model weights saved successfully.")

# Plot training and validation metrics
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Access accuracy using the correct key for the first output
    plt.plot(history.history['segmentation_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_segmentation_output_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_history(history)


