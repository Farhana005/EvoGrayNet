import glob

import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import os
import zipfile
from skimage.filters import threshold_otsu


# folder_path = r"E:\polyplocalizationdataset\PolypLocalizationDataset\PolypLocalizationDataset\CVC-ClinicDB\"
folder_path = r"E:\kvasir-seg\Kvasir-SEG\"  
# folder_path = r"E:\polyplocalizationdataset\PolypLocalizationDataset\PolypLocalizationDataset\CVC-ColonDB\"


def load_data(img_height, img_width, images_to_be_loaded, dataset):
    IMAGES_PATH = folder_path + 'images\'
    MASKS_PATH = folder_path + 'masks\'

    # Determine the file extension based on the dataset
    if dataset == 'kvasir':
        train_ids = glob.glob(IMAGES_PATH + "*.jpg")
    elif dataset in ['cvc-clinicdb', 'cvc-colondb', 'etis-laribpolypdb']:
        train_ids = glob.glob(IMAGES_PATH + "*.png")

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    # Define paths to save the resized images and masks
    resized_images_path = f"X_train_resized_{dataset}.npy"
    resized_masks_path = f"Y_train_resized_{dataset}.npy"

    # Load the resized data if it exists
    if os.path.exists(resized_images_path) and os.path.exists(resized_masks_path):
        print("Loading resized images and masks from disk...")
        X_train = np.load(resized_images_path)
        Y_train = np.load(resized_masks_path)
    else:
        # Initialize arrays for images and masks
        X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
        Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

        print('Resizing training images and masks: ' + str(images_to_be_loaded))
        for n, id_ in tqdm(enumerate(train_ids)):
            if n == images_to_be_loaded:
                break

            image_path = id_
            mask_path = image_path.replace("images", "masks")

            image = imread(image_path)
            mask_ = imread(mask_path)

            mask = np.zeros((img_height, img_width), dtype=np.bool_)

            # Resize image
            pillow_image = Image.fromarray(image)
            pillow_image = pillow_image.resize((img_height, img_width))
            image = np.array(pillow_image)

            X_train[n] = image / 255

            # Resize mask
            pillow_mask = Image.fromarray(mask_)
            pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
            mask_ = np.array(pillow_mask)

	    # Apply Otsu's adaptive threshold
	    threshold_value = threshold_otsu(mask_)
	    binary_mask = (mask_ >= threshold_value).astype(np.uint8)

            Y_train[n] = mask

        # Expand dimensions of Y_train
        Y_train = np.expand_dims(Y_train, axis=-1)

        # Save resized data to disk
        np.save(resized_images_path, X_train)
        np.save(resized_masks_path, Y_train)
        print("Resized images and masks saved to disk.")

    return X_train, Y_train

# Example usage
X, Y = load_data(img_size, img_size, -1, 'kvasir')
# Save data to a compressed zip file
with zipfile.ZipFile("/kaggle/working/resized_data_kvasir.zip", "w") as zipf:
    zipf.write(f"X_train_resized_kvasir.npy")
    zipf.write(f"Y_train_resized_kvasir.npy")

print("Resized data has been saved and zipped for download.")

# Splitting the data, seed for reproducibility

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle= True, random_state = seed_value)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.4, shuffle= True, random_state = seed_value)

x_train.shape, y_train.shape

type(x_train), len(x_train)

x_valid.shape, y_valid.shape

x_test.shape, y_test.shape

print(f"x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
print(f"x_valid shape: {x_valid.shape}, dtype: {x_valid.dtype}")
print(f"y_valid shape: {y_valid.shape}, dtype: {y_valid.dtype}")


y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')


print(f"x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
print(f"x_valid shape: {x_valid.shape}, dtype: {x_valid.dtype}")
print(f"y_valid shape: {y_valid.shape}, dtype: {y_valid.dtype}")



import cv2 as cv
import numpy as np

# Assuming x_train and y_train are already defined

x_train_augmented = np.empty((3 * x_train.shape[0], *x_train.shape[1:]))
y_train_augmented = np.empty((3 * y_train.shape[0], *y_train.shape[1:]))

for i in range(x_train.shape[0]):
    x_train_augmented[i] = x_train[i]

    # Horizontal flip
    x_flipped_horizontal = cv.flip(x_train[i], 1)
    x_train_augmented[i + x_train.shape[0]] = x_flipped_horizontal

    # Rotation by 90 degrees
    x_rotated_90 = np.rot90(x_train[i], k=1, axes=(1, 0))
    x_train_augmented[i + 2 * x_train.shape[0]] = x_rotated_90

    # Augment labels similarly
    y_train_augmented[i] = y_train[i]  # Original label

    # Horizontal flip for labels
    y_flipped_horizontal = cv.flip(y_train[i], 1)
    y_train_augmented[i + y_train.shape[0]] = (
        y_flipped_horizontal[:, :, np.newaxis] if y_flipped_horizontal.ndim == 2 else y_flipped_horizontal
    )

    # Rotation by 90 degrees for labels
    y_rotated_90 = np.rot90(y_train[i], k=1, axes=(1, 0))
    y_train_augmented[i + 2 * y_train.shape[0]] = (
        y_rotated_90[:, :, np.newaxis] if y_rotated_90.ndim == 2 else y_rotated_90
    )

# Now x_train_augmented and y_train_augmented contain the original and two augmented versions of the data


x_train_augmented.shape
