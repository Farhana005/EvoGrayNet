from data_loader import load_data
from model import build_model
from train import train_model
from test import evaluate_model

def main():
    # Load data
    print("Loading data...")
    X_train_augmented, Y_train_augmented, X_valid, Y_valid = load_data()

    # Build model
    print("Building model...")
    model = build_model()

    # Train model
    print("Training model...")
    train_model(model, X_train_augmented, Y_train_augmented, X_valid, Y_valid)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_valid, Y_valid)

if __name__ == "__main__":
    main()