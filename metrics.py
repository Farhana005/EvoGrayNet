import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to plot images
def plot_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
            ax.set_title(titles[i], fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    plt.show()

# Function to save images as PDF
def save_images_as_pdf(images, titles, rows, cols, filename="augmented_images.pdf"):
    with PdfPages(filename) as pdf:
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
                ax.set_title(titles[i], fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig)  # Save the figure to the PDF
        plt.close(fig)
    print(f"PDF saved as {filename}")

# Select only the 3rd (index 2) and 4th (index 3) images
selected_indices = [2, 5]
images = []
titles = []

for i in selected_indices:
    images.extend([
        x_train[i], y_train[i],  # Original image and label
        x_train_augmented[i + x_train.shape[0]], y_train_augmented[i + y_train.shape[0]],  # Flipped image and label
        x_train_augmented[i + 2 * x_train.shape[0]], y_train_augmented[i + 2 * y_train.shape[0]],  # Rotated image and label
    ])
    titles.extend([
        f"Original", f"Label",
        f"Flipped", f"Flipped Label",
        f"Rotated 90°", f"Rotated Label",
    ])

# Plot the images
plot_images(images, titles, rows=len(selected_indices), cols=6)

# Save as PDF
save_images_as_pdf(images, titles, rows=len(selected_indices), cols=6, filename="augmented_images.pdf")


import keras.backend as K
import tensorflow as tf


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)
    intersection = K.sum(predictions * ground_truth)
    union = K.sum(predictions) + K.sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

import keras.backend as K
import tensorflow as tf

def tversky_loss(ground_truth, predictions, alpha=0.7, smooth=1e-6):
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)

    true_positives = K.sum(predictions * ground_truth)
    false_positives = K.sum(predictions * (1 - ground_truth))
    false_negatives = K.sum((1 - predictions) * ground_truth)

    tversky = (true_positives + smooth) / (true_positives + alpha * false_positives + (1 - alpha) * false_negatives + smooth)

    return 1 - tversky

def dice_coef_from_logits(y_true, logits, eps=1e-7):
    probs = tf.sigmoid(logits)
    y_true = tf.cast(y_true, probs.dtype)
    inter = tf.reduce_sum(probs * y_true)
    denom = tf.reduce_sum(probs) + tf.reduce_sum(y_true)
    return (2.0 * inter + eps) / (denom + eps)
# import numpy as np
search_space = {
    # Gray Module
    "gray_filters": [32, 64,128],  # Controls model capacity and feature extraction power
    
    # ASPP Block
    "dilated_feature_extractor_rates": [[6, 12, 18], [12, 24, 36], [18, 36, 48]], # Essential for multi-scale context capture
    
    # Squeeze-Excitation
    "feature_recalibration_ratio": [2, 4, 8],    # Focuses on key feature channels

    #Dropout Rate
    "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],  # Regularization levels
    
    # Final Activation
    "final_activation": ['sigmoid', 'relu', 'tanh',  'softmax'],  # Chooses the right output for binary/multi-class segmentation
}



import random

def generate_random_architecture(search_space):
    """Generates a random architecture by sampling from the provided search space."""
    
    random_architecture = {
        "gray_filters": random.choice(search_space["gray_filters"]),
        "dilated_feature_extractor_rates": random.choice(search_space["dilated_feature_extractor_rates"]),
        "feature_recalibration_ratio": random.choice(search_space["feature_recalibration_ratio"]),
        # "network_depth": random.choice(search_space["network_depth"]),
        "dropout_rate": random.choice(search_space["dropout_rate"]),
        "final_activation": random.choice(search_space["final_activation"]),
    }
    
    return random_architecture 

population_size = 2
# Generate initial population directly
population = [generate_random_architecture(search_space) for _ in range(population_size)]

for i, architecture in enumerate(population):
    print(f"Architecture {i + 1}: {architecture}")


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, GlobalAveragePooling2D, Reshape, UpSampling2D, Concatenate
from tensorflow.keras.utils import plot_model

def dilated_feature_extractor(inputs, rates, dropout_rate):
    filters = inputs.shape[-1]

    aspp1 = Conv2D(filters, 3, dilation_rate=rates[0], padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    aspp1 = Dropout(dropout_rate)(aspp1)  # Dropout after the first dilation convolution
    
    aspp2 = Conv2D(filters, 3, dilation_rate=rates[1], padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    aspp2 = Dropout(dropout_rate)(aspp2)  # Dropout after the second dilation convolution
    
    aspp3 = Conv2D(filters, 3, dilation_rate=rates[2], padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    aspp3 = Dropout(dropout_rate)(aspp3)  # Dropout after the third dilation convolution

    global_avg_pool = GlobalAveragePooling2D()(inputs)
    global_avg_pool = Reshape((1, 1, filters))(global_avg_pool)
    global_avg_pool = Conv2D(filters, 1, activation='relu', kernel_initializer='he_normal')(global_avg_pool)
    global_avg_pool = UpSampling2D(size=(inputs.shape[1], inputs.shape[2]))(global_avg_pool)

    concatenated = Concatenate(axis=3)([aspp1, aspp2, aspp3, global_avg_pool])
    result = Conv2D(filters, 1, activation='relu', kernel_initializer='he_normal')(concatenated)

    return result

# Create a simple model for visualization
input_layer = Input(shape=(256, 256, 3))  # Example input shape
aspp_output = dilated_feature_extractor(input_layer, rates=[6, 12, 18], dropout_rate=0.1)
model = tf.keras.Model(inputs=input_layer, outputs=aspp_output)

# Save the model plot to a file accessible for download
plot_model(model, to_file="dilated_feature_extractor.png", show_shapes=True, show_layer_names=True)


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Activation, Multiply
from tensorflow.keras.utils import plot_model

def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, 1, padding='same', kernel_initializer='he_normal')(x)
    phi_g = Conv2D(inter_channels, 1, padding='same', kernel_initializer='he_normal')(g)
    add = Add()([theta_x, phi_g])
    relu = Activation('relu')(add)
    psi = Conv2D(1, 1, padding='same', kernel_initializer='he_normal')(relu)
    sigmoid = Activation('sigmoid')(psi)
    attn_out = Multiply()([x, sigmoid])
    return attn_out

# Model for visualization
x_input = Input(shape=(256, 256, 3), name="Input_X")
g_input = Input(shape=(256, 256, 3), name="Input_G")
output = attention_gate(x_input, g_input, inter_channels=16)
model = tf.keras.Model(inputs=[x_input, g_input], outputs=output)

# Save the model plot
plot_model(model, to_file="attention_gate.png", show_shapes=True, show_layer_names=True)


# model = model((256, 256, 3))
# model.summary()

import numpy as np
search_space = {
    # Gray Module
    "gray_filters": [16, 32, 64,128],  # Controls model capacity and feature extraction power
    
    # ASPP Block
    "dilated_feature_extractor_rates": [[6, 12, 18], [12, 24, 36], [18, 36, 48]], # Essential for multi-scale context capture
    
    # Squeeze-Excitation
    "feature_recalibration_ratio": [8, 16, 32],    # Focuses on key feature channels
    
    # Network Depth
    "network_depth": [5, 7, 9,11],                 # Determines how deeply the network processes the input
    
    #Dropout Rate
    "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],  # Regularization levels
    
    # Final Activation
    "final_activation": ['relu', 'tanh', 'sigmoid', 'softmax'],  # Chooses the right output for binary/multi-class segmentation
}






import random

def generate_random_architecture(search_space):
    """Generates a random architecture by sampling from the provided search space."""
    
    random_architecture = {
        "gray_filters": random.choice(search_space["gray_filters"]),
        "dilated_feature_extractor_rates": random.choice(search_space["dilated_feature_extractor_rates"]),
        "feature_recalibration_ratio": random.choice(search_space["feature_recalibration_ratio"]),
        # "network_depth": random.choice(search_space["network_depth"]),
        "dropout_rate": random.choice(search_space["dropout_rate"]),
        "final_activation": random.choice(search_space["final_activation"]),
    }
    
    return random_architecture 

population_size = 2
# Generate initial population directly
population = [generate_random_architecture(search_space) for _ in range(population_size)]

for i, architecture in enumerate(population):
    print(f"Architecture {i + 1}: {architecture}")


# print(y_train_augmented.shape)


print(np.unique(y_train))  # For binary, this should print [0, 1]


import numpy as np
import copy

def crossover_and_mutate(parent1, parent2, search_space, mutation_rate):
    """Performs crossover between two parent architectures and mutates the child architecture."""
    
    # Initialize an empty dictionary for the child architecture
    child = {}
    
    # Perform crossover by randomly selecting parameters from parents
    for key in parent1:
        if np.random.rand() < 0.5:  # 50% chance to select from either parent
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]

    # Apply mutation by randomly modifying architecture parameters
    for key in child:
        if np.random.rand() < mutation_rate:
            # Check if the key exists in the search space
            if key in search_space:
                if isinstance(search_space[key][0], list):  # For list-type parameters
                    # Mutate individual elements of the list
                    new_value = [np.random.choice(param) if np.random.rand() < mutation_rate else param 
                                 for param in child[key]]
                    child[key] = new_value
                else:
                    # Directly mutate the parameter by selecting a new value from the search space
                    child[key] = np.random.choice(search_space[key])

    return child


import matplotlib.pyplot as plt
import seaborn as sns
num_epochs=1

import numpy as np
# import tensorflow as tf
from keras import backend as K
from sklearn.metrics import auc

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + K.epsilon())
    return f1


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred, alpha=0.5):
    # Ensure y_true is float32
    y_true = K.cast(y_true, dtype='float32')
    
    bce_loss = K.binary_crossentropy(y_true, y_pred)
    dsc_loss = dice_loss(y_true, y_pred)
    return alpha * bce_loss + (1 - alpha) * dsc_loss
    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())



def iou(y_true, y_pred):

    # Convert predictions to binary class (0 or 1)
    y_pred = K.round(y_pred)  # Assuming y_pred is a probability map

    # Calculate true positives, false positives, false negatives
    true_positives = K.sum(y_true * y_pred)
    false_positives = K.sum((1 - y_true) * y_pred)
    false_negatives = K.sum(y_true * (1 - y_pred))

    # Calculate Intersection and Union
    intersection = true_positives
    union = true_positives + false_positives + false_negatives

    # Calculate mIoU
    iou = intersection / (union + K.epsilon())  # Adding epsilon to avoid division by zero

    return iou


import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed


# Plot training and validation accuracy and loss for each architecture as clusters
for gen_idx, generation in enumerate(gen_histories):  # Exclude the last summary entry
    print(f"\nGeneration {gen_idx + 1} Architectures:")
    
    for idx, (arch, fit, fps) in enumerate(zip(generation['population'], generation['fitness'], generation['fps'])):
        print(f"Architecture {idx + 1}: {arch} | Fitness: {fit[0]:.4f} | FPS: {fps:.2f}")
        
        # Retrieve the training history for plotting
        history = gen_histories[gen_idx]['history'][idx]
        
        # Plot training and validation accuracy as a scatter plot
        epochs = range(1, len(history['accuracy']) + 1)
        plt.scatter(epochs, history['accuracy'], label=f'Train Accuracy Gen {gen_idx + 1} Arch {idx + 1}', marker='+', alpha=0.7)
        plt.scatter(epochs, history['val_accuracy'], label=f'Val Accuracy Gen {gen_idx + 1} Arch {idx + 1}', marker='*', alpha=0.7)
    
    plt.title(f'Accuracy for Generation {gen_idx + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting FPS values for the current generation
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(generation['fps']) + 1), generation['fps'], marker='o', label='FPS', color='C2')
    plt.title(f'FPS for Generation {gen_idx + 1}')
    plt.xlabel('Architecture Index')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Saving generation histories to pickle files (if necessary)
for gen_idx, generation in enumerate(gen_histories):
    with open(f'generation_{gen_idx + 1}.pkl', 'wb') as f:
        pickle.dump(generation, f)

# Example usage of display_generation_histories
def load_generation_history(generation):
    """Load the generation history from a pickle file."""
    with open(f'generation_{generation}.pkl', 'rb') as f:
        return pickle.load(f)

def display_generation_histories(num_generations):
    """Load and display generation histories as a table."""
    records = []
    
    for gen in range(1, num_generations + 1):
        generation_data = load_generation_history(gen)
        for idx, (pop, fit) in enumerate(zip(generation_data['population'], generation_data['fitness'])):
            record = {
                'Generation': gen,
                'Model Index': idx + 1,
                'Gray Filters': pop['gray_filters'],
                'Dilation Feature Extractor Rates': pop['dilated_feature_extractor_rates'],
                'Feature Recalibration Ratio': pop['feature_recalibration_ratio'],
                # 'Network Depth': pop['network_depth'],
                'Final Activation': pop['final_activation'],
                'Fitness Accuracy': fit[0],
                'Fitness Loss': fit[1],
                'Validation Accuracy': fit[2],
                'Validation Loss': fit[3],
                'Execution Time (s)': generation_data['execution_time']
            }
            records.append(record)
    
    # Create a DataFrame from the records
    history_df = pd.DataFrame(records)
    print(history_df)

# Display the histories
display_generation_histories(num_generations)

# Plot all training and validation accuracies across all generations
all_train_accuracies = []
all_val_accuracies = []
all_fps = []

# Iterate through each generation and architecture to collect accuracies and FPS
for generation in gen_histories:  # Exclude the last summary entry
    for idx, (arch, fit) in enumerate(zip(generation['population'], generation['fitness'])):
        history = generation['history'][idx]
        all_train_accuracies.append(history['accuracy'])
        all_val_accuracies.append(history['val_accuracy'])
        all_fps.append(generation['fps'][idx])  # Collect FPS

# Convert lists to arrays for easier plotting
all_train_accuracies = np.array(all_train_accuracies)
all_val_accuracies = np.array(all_val_accuracies)
all_fps = np.array(all_fps)

# Plot all training accuracies
plt.figure(figsize=(12, 6))
for i in range(len(all_train_accuracies)):
    plt.plot(all_train_accuracies[i], color='C0', alpha=0.5)  # Training accuracy with some transparency
for i in range(len(all_val_accuracies)):
    plt.plot(all_val_accuracies[i], color='C1', alpha=0.5)  # Validation accuracy with some transparency

# Calculate and plot mean accuracy
mean_train_accuracy = all_train_accuracies.mean(axis=0)
mean_val_accuracy = all_val_accuracies.mean(axis=0)
plt.plot(mean_train_accuracy, color='C0', label='Mean Training Accuracy', linewidth=2)
plt.plot(mean_val_accuracy, color='C1', label='Mean Validation Accuracy', linewidth=2)

# Plotting settings
plt.title('Combined Training and Validation Accuracy Across All Generations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot FPS across all generations
plt.figure(figsize=(12, 6))
plt.plot(all_fps, marker='o', color='C2', alpha=0.5, label='FPS')
plt.title('FPS Across All Generations')
plt.xlabel('Architecture Index')
plt.ylabel('FPS')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Create separate scatter plots for training and validation accuracy
plt.figure(figsize=(12, 12)) 

# Scatter plot for Training Accuracy
plt.subplot(2, 1, 1)
for gen_idx, generation in enumerate(gen_histories):  # Exclude the last summary entry
    for idx, (arch, fit) in enumerate(zip(generation['population'], generation['fitness'])):
        # Retrieve the training history for plotting
        history = generation['history'][idx]
        
        # Scatter plot for training accuracy
        plt.scatter(range(len(history['accuracy'])), history['accuracy'],
                    marker='+' if gen_idx == 0 else '*',
                    label=f'Gen {gen_idx + 1} Arch {idx + 1}' if idx == 0 else "", 
                    alpha=0.7)  # Training accuracy

plt.title('Training Accuracy Across Generations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Scatter plot for Validation Accuracy
plt.subplot(2, 1, 2)
for gen_idx, generation in enumerate(gen_histories):  # Exclude the last summary entry
    for idx, (arch, fit) in enumerate(zip(generation['population'], generation['fitness'])):
        # Retrieve the validation history for plotting
        history = generation['history'][idx]
        
        # Scatter plot for validation accuracy
        plt.scatter(range(len(history['val_accuracy'])), history['val_accuracy'],
                    marker='o' if gen_idx == 0 else 's',
                    label=f'Gen {gen_idx + 1} Arch {idx + 1}' if idx == 0 else "", 
                    alpha=0.7)  # Validation accuracy

plt.title('Validation Accuracy Across Generations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()




for gen_idx, generation in enumerate(gen_histories):
    print(f"Generation {gen_idx + 1} Fitness Values: {generation['fitness']}")


# Find the generation and architecture with the highest validation accuracy
highest_val_acc = 0
best_gen_idx = None
best_arch_idx = None
best_architecture = None

for gen_idx, generation in enumerate(gen_histories):
    for idx, (arch, fit) in enumerate(zip(generation['population'], generation['fitness'])):
        val_accuracy = fit[2]  # Assuming this is the validation accuracy
        if val_accuracy > highest_val_acc:
            highest_val_acc = val_accuracy
            best_gen_idx = gen_idx
            best_arch_idx = idx
            best_architecture = arch  # Store the best architecture details

# Print the best architecture's details
print(f"Highest Validation Accuracy: {highest_val_acc:.4f}")
print(f"Best Generation: {best_gen_idx + 1}")
print(f"Best Architecture in Generation {best_gen_idx + 1}: {best_arch_idx + 1}")
print(f"Architecture Details: {best_architecture}")


# Initialize variables to store the best fitness scores and their corresponding architectures
best_training_fitness = 0
best_training_architecture = None

best_validation_fitness = 0
best_validation_architecture = None

# Iterate through each generation
for gen_idx, generation in enumerate(gen_histories):
    # Retrieve fitness values for each architecture
    for arch_idx, (arch, fitness) in enumerate(zip(generation['population'], generation['fitness'])):
        training_fitness = fitness[0]  # Assuming the first value is training accuracy
        validation_fitness = fitness[2]  # Assuming the third value is validation accuracy
        
        # Update the best training fitness if the current one is higher
        if training_fitness > best_training_fitness:
            best_training_fitness = training_fitness
            best_training_architecture = arch  # Store the architecture details
            
        # Update the best validation fitness if the current one is higher
        if validation_fitness > best_validation_fitness:
            best_validation_fitness = validation_fitness
            best_validation_architecture = arch  # Store the architecture details

# Print the overall best fitness scores and their corresponding architectures
print(f"Overall Best Training Fitness: {best_training_fitness:.4f}")
print(f"Best Training Architecture: {best_training_architecture}")

print(f"Overall Best Validation Fitness: {best_validation_fitness:.4f}")
print(f"Best Validation Architecture: {best_validation_architecture}")


def predict_in_batches(model, x_test, batch_size=2):
    num_batches = len(x_test) // batch_size
    all_predictions = []
    for i in range(num_batches):
        x_batch = x_test[i * batch_size:(i + 1) * batch_size]
        batch_predictions = model.predict(x_batch)
        all_predictions.append(batch_predictions)
    
    # Concatenate all predictions into a single array
    return np.concatenate(all_predictions, axis=0)

predictions = predict_in_batches(model, x_test, batch_size=2)


# If you want to visualize a few predictions
import matplotlib.pyplot as plt

# Show a few test images, ground truth, and predicted masks
num_images_to_show = 5
for i in range(num_images_to_show):
    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[i])  # assuming x_test contains images
    plt.title("Original Image")
    plt.axis('off')

    # Ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i])  # assuming y_test contains ground truth masks
    plt.title("Ground Truth")
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i], cmap='red')  # Predicted mask (assuming binary output)
    plt.title("Predicted Mask")
    plt.axis('off')
    plt.savefig("Kvasir.pdf", format="pdf")  # Save the plot as a PDF

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.measure import find_contours

# One-hot encode predicted masks
def one_hot_encode_mask(mask, num_classes=2):
    one_hot_mask = np.zeros((*mask.shape, num_classes))
    for c in range(num_classes):
        one_hot_mask[..., c] = (mask == c).astype(np.float64)
    return one_hot_mask

# Assuming predictions is the binary predicted mask
predictions_one_hot = one_hot_encode_mask((predictions > 0.5).astype(int), num_classes=2)

# Specify the number of images to visualize and save
num_images_to_show = 5
pdf_filename = "Kvasir_Predictions_OneHot_with_Red_Edges.pdf"

# Open a PDF for saving multiple pages
with PdfPages(pdf_filename) as pdf:
    for i in range(num_images_to_show):
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(x_test[i])  # Assuming x_test contains images
        plt.title("Original Image")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[i], cmap='gray')  # Assuming y_test contains binary masks
        plt.title("Ground Truth")
        plt.axis('off')

        # Predicted mask with red edge
        plt.subplot(1, 3, 3)
        plt.imshow(x_test[i])  # Display the original image as background
        plt.title("Predicted Mask with Edge")
        plt.axis('off')

        # Overlay the predicted mask edge
        binary_predicted_mask = (predictions_one_hot[i, ..., 1] > 0.5).astype(np.float64)  # Tumor class (1)
        contours = find_contours(binary_predicted_mask.squeeze(), level=0.5)  # Ensure mask is 2D
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # Red edges for mask

        # Save the current figure to the PDF
        pdf.savefig()  # Add the current plot as a page in the PDF

        # Show the current figure
        plt.show()

        # Close the figure to save memory
        plt.close()

# Print confirmation
print(f"Visualization PDF saved as {pdf_filename}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.measure import find_contours
from matplotlib.colors import ListedColormap

# Specify the number of images to visualize and save
num_images_to_show = 5
pdf_filename = "Kvasir_Predictions_with_Highlighted_Masks.pdf"

# Define a custom colormap for the mask highlight (e.g., green with transparency)
highlight_color = np.array([[0, 1, 0, 0.5]])  # RGBA: green with 50% transparency
highlight_cmap = ListedColormap(highlight_color)

# Open a PDF for saving multiple pages
with PdfPages(pdf_filename) as pdf:
    for i in range(num_images_to_show):
        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(x_test[i])  # Assuming x_test contains images
        plt.title("Original Image")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[i], cmap='gray')  # Assuming y_test contains binary masks
        plt.title("Ground Truth")
        plt.axis('off')

        # Predicted mask with highlight
        plt.subplot(1, 3, 3)
        plt.imshow(x_test[i])  # Display the original image as background
        plt.title("Predicted Mask")
        plt.axis('off')

        # Overlay the predicted mask area with highlight color
        binary_predicted_mask = (predictions[i] > 0.5).astype(np.float64)  # Binary mask (thresholded)
        plt.imshow(binary_predicted_mask, cmap=highlight_cmap, interpolation='none', alpha=0.5)

        # Overlay the predicted mask edges
        contours = find_contours(binary_predicted_mask.squeeze(), level=0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # Red edges for mask

        # Save the current figure to the PDF
        pdf.savefig()  # Add the current plot as a page in the PDF

        # Show the current figure
        plt.show()

        # Close the figure to save memory
        plt.close()

# Print confirmation
print(f"Visualization PDF saved as {pdf_filename}")
