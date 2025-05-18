import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import json
# tf.config.run_functions_eagerly(False)

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
# tf.profiler.experimental.stop()

def calculate_flops(model, input_shape=(256, 256, 3)):
    # Generate random input data based on the input shape
    inputs = tf.random.normal([1, *input_shape])  # A batch of one sample

    # Enable profiling to calculate FLOPS
    flops = None
    try:
        tf.profiler.experimental.start('/tmp/logdir')
        model(inputs)  # Run a forward pass to collect profile data
        tf.profiler.experimental.stop()
        
        # Approximate FLOPS calculation based on parameter count
        flops = model.count_params() * 2  # This is a simplified estimate
        print(f"Estimated FLOPS: {flops}")
        
    except tf.errors.AlreadyExistsError:
        print("Profiler already running, skipping start.")
    except tf.errors.UnavailableError:
        print("No profiler was active to stop.")
        
    return flops

# Define the fitness function
def fitness(random_architecture, x_train_augmented, y_train_augmented, x_valid, y_valid, epochs, batch_size, learning_rate):
    model = build_model(random_architecture)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Get trainable and non-trainable parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")
    
    # Compile the model with the specified learning rate
    model.compile(optimizer=optimizer, 
                  loss=combined_loss, 
                  metrics=['accuracy', sensitivity, precision, f1, dice_coefficient, iou])
    
    # Define input shape based on x_train_augmented
    input_shape = (None, *x_train_augmented.shape[1:])

    # Calculate and print FLOPS
    flops = calculate_flops(model, input_shape=(256, 256, 3))
    print(f"Estimated FLOPS for the model: {flops}")

    # Save FLOPS data to JSON for future reference
    flops_data = {
        'architecture': str(random_architecture),
        'flops': flops,
    }
    with open('flops_data.json', 'a') as f:
        json.dump(flops_data, f)
        f.write("\n")

    # Define paths for model and training history
    model_load_path = '/kaggle/input/kvasirweight/best_model_kvasir.h5'
    history_file = 'training_history_kvasir.json'
    model_save_path='best_model_kvasir.h5'

    # Initialize initial_epoch to 0
    initial_epoch = 0
    
    # Check if model weights and training history file exist to resume training
    if os.path.exists(model_save_path):
        try:
            model.load_weights(model_load_path)
            print("Resuming training with saved weights.")
        except ValueError as e:
            print(f"Could not load weights: {e}. Initializing new weights.")
    
     # epochs = 50  # Set this to your desired number of training epochs

    # Load epoch information if it exists
    initial_epoch = 0  # Default to starting from the beginning
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history_data = json.load(f)
            if 'epochs_completed' in history_data:
                completed_epochs = history_data['epochs_completed']
                if completed_epochs < epochs:
                    # Resume from the next epoch if training isn't completed
                    initial_epoch = completed_epochs
                    print(f"Resuming from epoch {initial_epoch + 1}")
                else:
                    print("All epochs completed; starting fresh.")
    


    # Set up callbacks
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_dice_coefficient', save_best_only=True, save_weights_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_dice_coefficient', patience=10, min_delta=0.001, mode='max', restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.1, patience=7, min_lr=1e-6, mode='max', verbose=0)

    # Train the model starting from the initial epoch
    history = model.fit(
        x_train_augmented,
        y_train_augmented,
        validation_data=(x_valid, y_valid),
        epochs=epochs,                # Total epochs to train
        initial_epoch=initial_epoch,   # Start from the last completed epoch
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint, reduce_lr,]
    )

    # Save the number of completed epochs to the history file
    completed_epochs = history.epoch[-1] + 1
    with open(history_file, 'w') as f:
        json.dump({'epochs_completed': completed_epochs}, f)

    # Extract metrics from the training history
    max_val_accuracy = max(history.history['val_accuracy'])
    final_val_loss = history.history['val_loss'][-1]
    final_train_accuracy = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]

    return max_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss, history


import os
import time
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Initialize parameters
population_size = 2  # Set your desired population size
num_generations = 5  # Set the number of generations
mutation_rate = 0.1
population = [generate_random_architecture(search_space) for _ in range(population_size)]

gen_histories = []

random.seed(10)
# Evolutionary process
total_iterations = num_generations * population_size
# Set up MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

# Define your model and compile it within the strategy scope
with strategy.scope():
    
    for gen in range(num_generations):
        print(f"\n===== Generation {gen + 1}/{num_generations} =====")

        # Measure time taken per generation
        start_time = time.time()

        # Evaluate the fitness of each architecture with progress tracking
        fitness_values = []
        generation_histories = []  # Store individual architecture histories for this generation
        fps_values = []  # Store FPS for each architecture

        # Evaluate fitness for each architecture in the population
        for i, random_architecture in enumerate(tqdm(population, desc=f"Evaluating generation {gen + 1}", leave=False)):
            print(f"===== Population {i + 1}/{population_size} =====")

            # Print the selected architecture
            print(f"Selected Architecture {i + 1}: {random_architecture}") 

            max_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss, history = fitness(
                random_architecture, x_train_augmented, y_train_augmented, x_valid, y_valid, epochs=20, batch_size=6, learning_rate=0.001
            )

            # Calculate FPS based on the elapsed time
            training_end_time = time.time()
            elapsed_time = training_end_time - start_time
            fps = 2 / elapsed_time if elapsed_time > 0 else 0  # Assuming max 2 epochs
            fps_values.append(fps)
            
            

            # Store the fitness values
            fitness_val = (max_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss)
            fitness_values.append(fitness_val)

            print(f"  - Architecture {i + 1}/{population_size} fitness (accuracy): {fitness_val[0]:.4f} | FPS: {fps:.2f}")

            # Store the training history for plotting later
            generation_histories.append(history.history)

        # Generate new population through crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            # Select parents based on fitness (here simply random selection for illustration)
            parent1, parent2 = random.choices(population, k=2)
            child = crossover_and_mutate(parent1, parent2, search_space, mutation_rate)
            new_population.append(child)

        population = new_population  # Update population to new population

        # Record generation history
        gen_histories.append({
            'population': population,   # Store the current population
            'fitness': fitness_values,  # Store the fitness values
            'fps': fps_values,          # Store the FPS values
            'execution_time': time.time() - start_time,  # Store the execution time
            'history': generation_histories  # Store the histories of this generation
        })

        # Show generation summary
        print(f"\nSummary of Generation {gen + 1}:") 
        print(f"  - Best fitness: {max(fitness_values, key=lambda x: x[0])[0]:.4f}")
        print(f"  - Time taken: {gen_histories[-1]['execution_time']:.2f} seconds")

# Initialize variables to track the best architecture and its fitness
best_fitness = 0  # Initialize to a low value
best_architecture = None  # Initialize to None

# Iterate through each generation in gen_histories
for generation in gen_histories:
    # Retrieve the population and corresponding fitness scores
    architectures = generation['population']
    fitness_scores = generation['fitness']
    
    # Iterate through architectures and their fitness scores
    for architecture, fitness in zip(architectures, fitness_scores):
        # Assuming you're interested in the first fitness score (e.g., accuracy)
        accuracy = fitness[0]
        if accuracy > best_fitness:  # Compare accuracy to find the best
            best_fitness = accuracy  # Update the best fitness score
            best_architecture = architecture  # Store the best architecture

# Now 'best_architecture' holds the best architecture across all generations
print(f"Best Architecture: {best_architecture} with Fitness: {best_fitness}")

# Build the final model using the best architecture found
final_model = build_model(best_architecture)


import time

# Step 1: Define and compile the best model
final_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=dice_coefficient_loss, 
                  metrics=['accuracy', sensitivity, specificity, precision, f1, dice_coefficient])

# Start measuring the execution time
start_time = time.time()

# Step 2: Train the best model
history = final_model.fit(x_train_augmented, y_train_augmented, validation_data=(x_valid, y_valid), epochs=10, batch_size=16)

# Step 3: Evaluate the best model on the validation set
loss, accuracy, sensitivity_value, specificity_value, precision_value, f1_value, dice_coef = final_model.evaluate(x_valid, y_valid, verbose=1)

# Stop measuring the execution time
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")


# Print training performance metrics
print("Training Loss:", history.history['loss'])
print("Training Accuracy:", history.history['accuracy'])
print("Training Precision:", history.history['precision'])
print("Training Sensitivity:", history.history['sensitivity'])
print("Training Specificity:", history.history['specificity'])
print("Training F1:", history.history['f1'])
print("Training Dice Coefficient:", history.history['dice_coefficient'])



print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
print("Validation Precision:", precision_value)
print("Validation Sensitivity:", sensitivity_value)
print("Validation Specificity:", specificity_value)
print("Validation F1:", f1_value)
print("Validation Dice Coefficient:", dice_coef)





print(f"x_train_augmented dtype: {x_train_augmented.dtype}")
print(f"y_train_augmented dtype: {y_train_augmented.dtype}")
print(f"x_valid dtype: {x_valid.dtype}")
print(f"y_valid dtype: {y_valid.dtype}")
