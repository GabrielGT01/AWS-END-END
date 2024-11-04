# | filename: script_cv.py
# | code-line-numbers: true

import argparse
import json
import os
import tarfile
from pathlib import Path
from comet_ml import Experiment
from math import sqrt 

import keras
from keras.regularizers import l2
import numpy as np
import pandas as pd
from keras import Input
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from packaging import version
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

def train(
    model_directory,
    train_path,
    validation_path,
    pipeline_path,
    experiment,
    epochs,
    batch_size=16,
    learning_rate=0.001
):
    print(f"Keras version: {keras.__version__}")

    # Load and concatenate training and validation datasets
    X_train = pd.read_csv(Path(train_path) / "train.csv", header=None)
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)
    
    X_validation = pd.read_csv(Path(validation_path) / "validation.csv", header=None)
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation = X_validation.drop(X_validation.columns[-1], axis=1)

    # Combine training and validation data
    X = pd.concat([X_train, X_validation], axis=0, ignore_index=True)
    y = pd.concat([y_train, y_validation], axis=0, ignore_index=True)

    print("X shape:", X.shape)
    print("Input dimensions:", X.shape[1])

    # Compute balanced class weights
    #class_weights = compute_class_weight(
        #class_weight='balanced',
        #classes=np.unique(y),
        #y=y
    # )
    #class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    # Extra boost for waitlist class

    #class_weight_dict[2] *= 2.0  # Waitlist
    #class_weight_dict[0] *= 1.5  # Admit

    # Define the 85% reduced number of samples for each class 70% train and 15% validation data, so 85
    admit_samples_85 = 765  # class 0
    deny_samples_85 = 4415  # class 1
    waitlist_samples_85 = 85  # class 2
    
    # Max samples (Deny class)
    max_samples = deny_samples_85
    
    # Calculate the class weights using Inverse Frequency Weighting
    class_weight_dict = {
        0: max_samples / admit_samples_85,      # Admit: 4414.9 / 765
        1: max_samples / deny_samples_85,       # Deny: 4414.9 / 4414.9
        2: max_samples / waitlist_samples_85 *2.0    # Waitlist: 4414.9 / 85
    }

    # Initialize StratifiedKFold for better class distribution
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Save the final model
    model_filepath = (
        Path(model_directory) / "001"
        if version.parse(keras.__version__) < version.parse("3")
        else Path(model_directory) / "admission.keras"
    )

    # List to store accuracy for each fold
    fold_accuracies = []

    # Perform stratified cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining on fold {fold}")
        
        # Split the data into training and validation sets for this fold
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Create the enhanced model
        model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(3, activation="softmax"),
        ])

        # Compile with Adam optimizer and gradient clipping
        model.compile(
            optimizer=RMSprop(learning_rate=learning_rate, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",  
            metrics=["accuracy"]
        )

        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_filepath,
                monitor='val_accuracy',
                save_best_only=True,
                mode = 'max'
            )
        ]

        # Train the model
        model.fit(
            X_train_fold,
            y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=2
        )

        # Evaluate the model on the validation set for this fold
        val_predictions = np.argmax(model.predict(X_val_fold), axis=-1)
        fold_accuracy = accuracy_score(y_val_fold, val_predictions)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold} validation accuracy: {fold_accuracy}")

    # Calculate mean validation accuracy across all folds
    mean_val_accuracy = np.mean(fold_accuracies)
    print(f"\nMean validation accuracy using cross-validation: {mean_val_accuracy}")

    # Retrain the model on the entire dataset
    print("\nTraining final model on entire dataset")
    final_model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(3, activation="softmax"),
            ])

    final_model.compile(
        optimizer=RMSprop(learning_rate=learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",  
        metrics=["accuracy"]
    )

    # Train the final model with a validation split
    history = final_model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        validation_split=0.2,  # Use 20% of data for validation
        callbacks=callbacks,
        verbose=1
    )

    # Save the final model using model.save()
    final_model.save(model_filepath)

    # Evaluate the model on the entire dataset again
    print("\nEvaluating final model on entire dataset")
    final_predictions = np.argmax(final_model.predict(X), axis=-1)
    final_accuracy = accuracy_score(y, final_predictions)
    final_precision = precision_score(y, final_predictions, average='weighted')
    final_recall = recall_score(y, final_predictions, average='weighted')

    print(f"\nFinal model metrics:")
    print(f"Final model accuracy on entire dataset: {final_accuracy:.4f}")
    print(f"Final model precision on entire dataset: {final_precision:.4f}")
    print(f"Final model recall on entire dataset: {final_recall:.4f}")

    # Save the transformation pipelines inside the model directory
    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    # Log metrics to Comet.ml if experiment is set up
    if experiment:
        experiment.log_parameters({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "class_weights": class_weight_dict
        })
        experiment.log_dataset_hash(X)
        experiment.log_confusion_matrix(
            y.astype(int), final_predictions.astype(int)
        )
        experiment.log_model("admission", model_filepath.as_posix())
        experiment.log_metric("mean_cv_accuracy", mean_val_accuracy)
        experiment.log_metric("final_model_accuracy", final_accuracy)
        experiment.log_metric("final_model_precision", final_precision)
        experiment.log_metric("final_model_recall", final_recall)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    args, _ = parser.parse_known_args()

    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None
    )

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", "{}"))
    job_name = training_env.get("job_name", None) if training_env else None

    if job_name and experiment:
        experiment.set_name(job_name)

    train(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
