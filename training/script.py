# | filename: script.py
# | code-line-numbers: true

import argparse
import json
import os
import tarfile
from pathlib import Path
from comet_ml import Experiment

import keras
import numpy as np
import pandas as pd
from math import sqrt 

from keras import Input
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential  # Removed trailing comma
from keras.optimizers import Adam, SGD
from packaging import version
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

def train(
    model_directory,
    train_path,
    validation_path,
    pipeline_path,
    experiment,
    epochs=50,
    batch_size=16,
    learning_rate=0.001,  # Added missing learning_rate parameter
    ):
    
    print(f"Keras version: {keras.__version__}")

    # load the train from the preprocessing step
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation = X_validation.drop(X_validation.columns[-1], axis=1)

    print(f"Input shape for training: {X_train.shape}")
    print(f"Input shape for validation: {X_validation.shape}")

    # Define class weights - Fixed undefined 'y' variable
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),  # Changed y to y_train
        y=y_train  # Changed y to y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    class_weight_dict[2] *= 1.5

    model_filepath = (
        Path(model_directory) / "001"
        if version.parse(keras.__version__) < version.parse("3")
        else Path(model_directory) / "admission.keras"
    )
    
    # Build model - Fixed undefined X variable
    model = Sequential([
            Input(shape=(X_train.shape[1],)),  # Changed X to X_train
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.15),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dense(3, activation="softmax"),
        ])

    model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_filepath,
                monitor='val_loss',
                save_best_only=True
            )
        ]

    # Fixed variable names
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,  # Fixed variable name
        callbacks=callbacks,  # Fixed variable name
        verbose=1,
    )

    model.save(model_filepath)

    predictions = np.argmax(model.predict(X_validation), axis=-1)

    val_accuracy = accuracy_score(y_validation, predictions)
    val_precision = precision_score(y_validation, predictions, average='weighted')
    val_recall = recall_score(y_validation, predictions, average='weighted')

    print(f"Validation accuracy: {val_accuracy}")
    print(f"Validation precision: {val_precision}")
    print(f"Validation recall: {val_recall}")

    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)
    
    if experiment:
        experiment.log_parameters(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
            }
        )
        experiment.log_dataset_hash(X_train)
        experiment.log_confusion_matrix(
            y_validation.astype(int), predictions.astype(int)
        )
        experiment.log_model("admission_model", model_filepath.as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)  # This argument is passed from the pipeline
    parser.add_argument("--batch_size", type=int, default=32)
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
        epochs=args.epochs,  # Pass epochs from args
        batch_size=args.batch_size,
        learning_rate = args.learning_rate
    )
