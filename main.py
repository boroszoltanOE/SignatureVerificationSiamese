import random
import cv2
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, Input, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.regularizers import l2

from sklearn.utils import shuffle
from tensorflow.keras import layers
from PIL import ImageOps, Image
from skimage.morphology import skeletonize

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy('mixed_float16'))

# Limit GPU memory growth to prevent TensorFlow from allocating all memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(gpu)
    except RuntimeError as e:
        print(e)

del gpus

datasets_path = './datasets'

path = os.path.join(datasets_path, 'CEDAR/CEDAR')
# path = os.path.join(datasets_path, 'BHSig260-Bengali/BHSig260-Bengali')
# path = os.path.join(datasets_path, 'BHSig260-Hindi/BHSig260-Hindi')

genuine_images = {}
forgery_images = {}

for directory in os.listdir(path):
    dir_path = os.path.join(path, directory)
    if os.path.isdir(dir_path):
        images = os.listdir(dir_path)
        images.sort()
        if 'CEDAR' in path:
            forgery_images[directory] = images[:24]  # First 24 signatures are forged
            genuine_images[directory] = images[24:]  # Next 24 signatures are genuine
        else:
            forgery_images[directory] = images[:30]  # First 30 signatures are forged
            genuine_images[directory] = images[30:]  # Next 24 signatures are genuine

del directory, dir_path, images

# Quick check to confirm we have data for all authors
print("Number of authors with genuine images:", len(genuine_images))
print("Number of authors with forgery images:", len(forgery_images))

# Sample output for one author
author = list(genuine_images.keys())[0]
print(f"Sample genuine images for author {author}:", genuine_images[author][:3])
print(f"Sample forgery images for author {author}:", forgery_images[author][:3])

del author

IMG_WIDTH, IMG_HEIGHT = 128, 192  # Set your image dimensions here


# Step 3: Full Preprocessing Function
def preprocessing(image, augment_percent=0.0, plot=None):
    def cropping(gray_image):
        # Blur and threshold to find non-white areas
        blur = cv2.GaussianBlur(gray_image, (25, 25), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return gray_image[y:y + h, x:x + w]
        return gray_image

    def apply_augmentation(image, padding=40):
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)

        # Convert the image to TensorFlow tensor and add channel dimension
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)  # Add channel dimension

        image = tf.ensure_shape(image, (None, None, 1))

        # Define the augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.05, interpolation='nearest'),
            layers.RandomBrightness(0.05),  # Randomly change the brightness of images
            layers.RandomZoom(0.05, interpolation='nearest'),  # Zoom with nearest-neighbor interpolation
        ])

        augmented_image = data_augmentation(image)
        augmented_image = tf.squeeze(augmented_image).numpy()  # Remove the channel dimension and convert back to NumPy

        return augmented_image

    steps = {}

    # Grayscale Conversion
    grayscale_image_np = np.array(ImageOps.grayscale(image))
    steps['Grayscale'] = grayscale_image_np

    # Apply augmentation to a certain percentage of images
    if random.random() < augment_percent:
        grayscale_image_np = apply_augmentation(grayscale_image_np)
        steps['Augmented'] = grayscale_image_np

    # Remove White Space
    cropped_image_np = cropping(grayscale_image_np.astype(np.uint8))
    steps['Cropped'] = cropped_image_np

    # Resizing (before skeletonization)
    resized_image = cv2.resize(cropped_image_np, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_LINEAR)
    steps['Resized'] = resized_image

    # Inversion and Binary Thresholding (combined)
    inverted_image = 255 - resized_image
    _, thresholded_image = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps['Inverted'] = inverted_image
    steps['Thresholded'] = thresholded_image

    # # Skeletonization (after augmentation, if any)
    # skeletonized_image = skeletonize(binary_image // 255)  # Skeletonize expects binary image
    # skeletonized_image = (skeletonized_image * 255).astype(np.uint8)  # Convert back to 0-255 scale
    # steps['Skeletonized'] = skeletonized_image

    # Normalization
    normalized_image = thresholded_image / 255.0
    steps['Normalized'] = normalized_image

    # Convert to TensorFlow tensor
    tensor_image = tf.convert_to_tensor(normalized_image, dtype=tf.float32)
    tensor_image = tf.expand_dims(tensor_image, axis=-1)

    # del grayscale_image_np, cropped_image_np, resized_image, inverted_image, thresholded_image, normalized_image

    del grayscale_image_np, cropped_image_np, resized_image, inverted_image, thresholded_image, normalized_image

    if plot:
        return steps
    else:
        return tensor_image


# Visualization Function
def visualize_preprocessing(image):
    steps = preprocessing(image, plot=True)

    titles = list(steps.keys())
    images = list(steps.values())

    fig, axes = plt.subplots(2, len(images), figsize=(20, 10))
    for i, (ax_img, ax_hist, img, title) in enumerate(zip(axes[0], axes[1], images, titles)):
        if isinstance(img, tf.Tensor):
            img = img.numpy().squeeze()  # Convert tensor to numpy and squeeze the channel dimension

        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(title)
        ax_img.axis('off')

        ax_hist.hist(img.ravel(), bins=256, color='black', alpha=0.75)
        ax_hist.set_title(f'{title} Histogram')

    plt.tight_layout()
    plt.show()

    del steps, titles, images, fig, axes, ax_img, ax_hist, img, title


# Example usage
author = list(genuine_images.keys())[10]
genuine_image_path = os.path.join(path, author, genuine_images[author][0])
genuine_image = Image.open(genuine_image_path)
visualize_preprocessing(genuine_image)

del genuine_image, genuine_image_path, author

# Split the authors to training, validating and testing sets

BATCH_SIZE = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.10
TEST_SPLIT = 0.20

# Assume genuine_images and forgery_images are dictionaries as described
authors = list(genuine_images.keys())
random.shuffle(authors)

# Split authors into train, validation, and test sets
num_authors = len(authors)
train_authors = authors[:int(TRAIN_SPLIT * num_authors)]
val_authors = authors[int(TRAIN_SPLIT * num_authors):int((TRAIN_SPLIT + VAL_SPLIT) * num_authors)]
test_authors = authors[int((TRAIN_SPLIT + VAL_SPLIT) * num_authors):]

del authors, num_authors


def create_pairs(authors_list):
    genuine_pairs, genuine_labels = [], []
    forgery_pairs, forgery_labels = [], []
    for author in authors_list:
        genuine_list = genuine_images[author]
        forgery_list = forgery_images[author]

        # Generate positive pairs (genuine-genuine)
        for i in range(len(genuine_list)):
            for j in range(i + 1, len(genuine_list)):
                genuine_pairs.append(
                    (os.path.join(path, author, genuine_list[i]), os.path.join(path, author, genuine_list[j])))
                genuine_labels.append(1)

        # Generate negative pairs (genuine-forgery)
        for i in range(len(genuine_list)):
            for j in range(len(forgery_list)):
                forgery_pairs.append(
                    (os.path.join(path, author, genuine_list[i]), os.path.join(path, author, forgery_list[j])))
                forgery_labels.append(0)

        del genuine_list, forgery_list

    if len(genuine_pairs) < len(forgery_pairs):
        forgery_pairs = forgery_pairs[:len(genuine_pairs)]
        forgery_labels = forgery_labels[:len(genuine_labels)]
    elif len(genuine_pairs) > len(forgery_pairs):
        genuine_pairs = genuine_pairs[:len(forgery_pairs)]
        genuine_labels = genuine_labels[:len(forgery_labels)]
    else:
        pass

    pairs = genuine_pairs + forgery_pairs
    labels = genuine_labels + forgery_labels

    del genuine_pairs, genuine_labels, forgery_pairs, forgery_labels

    return shuffle(pairs, labels)


train_pairs, train_labels = create_pairs(train_authors)
val_pairs, val_labels = create_pairs(val_authors)
test_pairs, test_labels = create_pairs(test_authors)

del train_authors, val_authors, test_authors

print("-------------------------")

print("[TRAINING INFO]")
print("Number of pairs:", len(train_pairs), " example: ", train_pairs[0])
print("Number of labels:", len(train_labels), " example: ", train_labels[0])

print("-------------------------")

print("[VALIDATON INFO]")
print("Number of pairs:", len(val_pairs), " example: ", val_pairs[0])
print("Number of labels:", len(val_labels), " example: ", val_labels[0])

print("-------------------------")

print("[TEST INFO]")
print("Number of pairs:", len(test_pairs), " example: ", test_pairs[0])
print("Number of labels:", len(test_labels), " example: ", test_labels[0])

print("-------------------------")


def batch_generator(all_pairs, all_labels, batch_size, img_h, img_w):
    while True:  # Keep generating batches indefinitely
        for start in range(0, len(all_pairs), batch_size):
            end = min(start + batch_size, len(all_pairs))
            current_batch_size = end - start

            pairs = [np.zeros((current_batch_size, img_w, img_h, 1)) for _ in range(2)]
            targets = np.zeros((current_batch_size,))

            for i in range(current_batch_size):
                img1 = preprocessing(Image.open(all_pairs[start + i][0]))
                img2 = preprocessing(Image.open(all_pairs[start + i][1]))

                pairs[0][i, :, :, :] = img1
                pairs[1][i, :, :, :] = img2
                targets[i] = all_labels[start + i]

            yield pairs, targets


train_gen = batch_generator(train_pairs, train_labels, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
val_gen = batch_generator(val_pairs, val_labels, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
test_gen = batch_generator(test_pairs, test_labels, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

# Get a single batch from the train generator
train_batch, train_labels = next(train_gen)

# Check the shape of the images and labels
print(f"Shape of img1 batch: {train_batch[0].shape}")  # Should be (batch_size, img_h, img_w, 1)
print(f"Shape of img2 batch: {train_batch[1].shape}")  # Should be (batch_size, img_h, img_w, 1)
print(f"Shape of labels: {train_labels.shape}")  # Should be (batch_size,)


# Visualize some images from the batch
def visualize_batch(images1, images2, labels, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    for i in range(num_samples):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax1.imshow(images1[i].squeeze(), cmap='gray')
        ax2.imshow(images2[i].squeeze(), cmap='gray')
        ax1.set_title(f"Pair {i + 1} - Image 1 (Label: {labels[i]})")
        ax2.set_title(f"Pair {i + 1} - Image 2 (Label: {labels[i]})")
        ax1.axis('off')
        ax2.axis('off')
    plt.tight_layout()
    plt.show()


# Visualize the first 5 samples in the batch
visualize_batch(train_batch[0], train_batch[1], train_labels, num_samples=5)


def euclidean_distance(vects):
    """
        Compute Euclidean Distance between two vectors
    """
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    """
        Contrastive loss function
    """
    margin = 1.0
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(y_true, y_pred, threshold=0.5):
    """
        Compute accuracy based on a distance threshold
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


def create_base_network_signet(input_shape, alpha_value=0.1, l2_value=1e-3, dropout_value=0.5):
    """
        Base Siamese Network
    """
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=l2(l2_value)))
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(l2_value)))
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(l2_value)))
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, kernel_regularizer=l2(l2_value)))  # Feature vector
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(Dropout(rate=dropout_value))

    model.add(Dense(128, kernel_regularizer=l2(l2_value)))  # Feature vector
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(Dropout(rate=dropout_value))

    return model


INPUT_SHAPE = (128, 192, 1)

# network definition
base_network = create_base_network_signet(INPUT_SHAPE)

input_a = Input(shape=INPUT_SHAPE)
input_b = Input(shape=INPUT_SHAPE)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the Euclidean distance between the two vectors in the latent space
distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

# Define the Siamese Network model
siamese_model = Model(inputs=[input_a, input_b], outputs=distance)

rms = RMSprop()

# Compile the model
siamese_model.compile(optimizer=rms, loss=contrastive_loss, metrics=[compute_accuracy])

# Model summary
siamese_model.summary()

train_steps = len(train_pairs) // BATCH_SIZE
val_steps = len(val_pairs) // BATCH_SIZE

siamese_model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=10
)

# Assuming val_pairs is a tuple of (val_image_a, val_image_b)
# and val_labels is the ground truth for validation pairs (1 for similar, 0 for dissimilar)

# Extract embeddings from the validation set
embeddings_a = base_network.predict(val_pairs[0])  # Embeddings for first image in pair
embeddings_b = base_network.predict(val_pairs[1])  # Embeddings for second image in pair


# Compute Euclidean distances between the embeddings
def compute_distances(embeddings_a, embeddings_b):
    return np.sqrt(np.sum(np.square(embeddings_a - embeddings_b), axis=1))


distances = compute_distances(embeddings_a, embeddings_b)

# Separate distances for positive and negative pairs
positive_distances = distances[val_labels == 1]  # Positive pairs (similar)
negative_distances = distances[val_labels == 0]  # Negative pairs (dissimilar)

plt.figure(figsize=(10, 6))
plt.hist(positive_distances, bins=30, alpha=0.5, label='Positive Pairs (Similar)', color='blue')
plt.hist(negative_distances, bins=30, alpha=0.5, label='Negative Pairs (Dissimilar)', color='red')
plt.title('Distance Distribution of Positive and Negative Pairs')
plt.xlabel('Euclidean Distance')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

# Now you can visually inspect the distribution and adjust your threshold
