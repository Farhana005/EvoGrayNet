import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json
import time
import random
import numpy as np
from tqdm import tqdm

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def calculate_flops(model, input_shape):
    """
    Returns total float ops (FLOPs) for one forward pass using TF profiler.
    input_shape: (D,H,W,C) for 3D or (H,W,C) for 2D (no batch dimension).
    """
    # Ensure model is built
    dummy = tf.zeros([1, *input_shape], dtype=tf.float32)
    _ = model(dummy, training=False)

    @tf.function
    def _forward(x):
        return model(x, training=False)

    concrete = _forward.get_concrete_function(
        tf.TensorSpec([1, *input_shape], tf.float32)
    )

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

    return int(flops.total_float_ops) if flops is not None else 0



# -------------------------
def fitness(random_architecture,
            x_train_augmented, y_train_augmented,
            x_valid, y_valid,
            epochs, batch_size, learning_rate):

    model = build_model(random_architecture)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Trainable / non-trainable params (same as yours)
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")

    # Compile (same as yours)
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['accuracy', sensitivity, precision, f1, dice_coefficient, iou]
    )


    inferred_input_shape = tuple(x_train_augmented.shape[1:])
    flops = 0
    try:
        flops = calculate_flops(model, input_shape=inferred_input_shape)
        print(f"Estimated FLOPs (profiler) for the model: {flops}")
    except Exception as e:
        print(f"[WARN] FLOPs profiling failed: {e}")
        flops = 0

    # Keep your FLOPs logging (same filename/format)
    flops_data = {
        'architecture': str(random_architecture),
        'input_shape': list(inferred_input_shape),
        'flops': int(flops),
    }
    with open('flops_data.json', 'a') as f:
        json.dump(flops_data, f)
        f.write("\n")

    
    early_stopping = EarlyStopping(
        monitor='val_dice_coefficient',
        patience=10,
        min_delta=0.001,
        mode='max',
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_dice_coefficient',
        factor=0.1,
        patience=7,
        min_lr=1e-6,
        mode='max',
        verbose=0
    )

    history = model.fit(
        x_train_augmented,
        y_train_augmented,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    # Extract metrics (same as yours)
    max_val_accuracy = max(history.history['val_accuracy'])
    final_val_loss = history.history['val_loss'][-1]
    final_train_accuracy = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]

    return max_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss, history




# Initialize parameters
population_size = 2
num_generations = 5
mutation_rate = 0.1
population = [generate_random_architecture(search_space) for _ in range(population_size)]

gen_histories = []

random.seed(10)

# Multi-GPU
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with strategy.scope():

    for gen in range(num_generations):
        print(f"\n===== Generation {gen + 1}/{num_generations} =====")
        start_time = time.time()

        fitness_values = []
        generation_histories = []
        fps_values = []

        for i, random_architecture in enumerate(tqdm(population, desc=f"Evaluating generation {gen + 1}", leave=False)):
            print(f"===== Population {i + 1}/{population_size} =====")
            print(f"Selected Architecture {i + 1}: {random_architecture}")

            max_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss, history = fitness(
                random_architecture,
                x_train_augmented, y_train_augmented,
                x_valid, y_valid,
                epochs=20, batch_size=6, learning_rate=0.001
            )

            training_end_time = time.time()
            elapsed_time = training_end_time - start_time
            fps = 2 / elapsed_time if elapsed_time > 0 else 0  # kept as-is (your code)
            fps_values.append(fps)

            fitness_val = (max_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss)
            fitness_values.append(fitness_val)

            print(f"  - Architecture {i + 1}/{population_size} fitness (accuracy): {fitness_val[0]:.4f} | FPS: {fps:.2f}")
            generation_histories.append(history.history)

        gen_histories.append({
            'population': population,
            'fitness': fitness_values,
            'fps': fps_values,
            'execution_time': time.time() - start_time,
            'history': generation_histories
        })

        # Generate new population (kept as-is: random parent selection)
        new_population = []
        scores = [fv[0] for fv in fitness_values]  # fv[0] = max_val_accuracy
        k = max(2, population_size // 2)
        top_idx = np.argsort(scores)[::-1][:k]
        selected = [population[j] for j in top_idx]
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover_and_mutate(parent1, parent2, search_space, mutation_rate)
            new_population.append(child)
        
        population = new_population

        print(f"\nSummary of Generation {gen + 1}:")
        print(f"  - Best fitness: {max(fitness_values, key=lambda x: x[0])[0]:.4f}")
        print(f"  - Time taken: {gen_histories[-1]['execution_time']:.2f} seconds")


# Pick best architecture across generations (same logic)
best_fitness = 0
best_architecture = None

for generation in gen_histories:
    architectures = generation['population']
    fitness_scores = generation['fitness']
    for architecture, fitness_tuple in zip(architectures, fitness_scores):
        accuracy = fitness_tuple[0]
        if accuracy > best_fitness:
            best_fitness = accuracy
            best_architecture = architecture

print(f"Best Architecture: {best_architecture} with Fitness: {best_fitness}")

# Build and train final model (unchanged)
final_model = build_model(best_architecture)

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=dice_coefficient_loss,
    metrics=['accuracy', sensitivity, specificity, precision, f1, dice_coefficient]
)

start_time = time.time()
history = final_model.fit(
    x_train_augmented, y_train_augmented,
    validation_data=(x_valid, y_valid),
    epochs=10,
    batch_size=16
)

loss, accuracy, sensitivity_value, specificity_value, precision_value, f1_value, dice_coef = final_model.evaluate(
    x_valid, y_valid, verbose=1
)

execution_time = time.time() - start_time
print("Execution Time:", execution_time, "seconds")

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
