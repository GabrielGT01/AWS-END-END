# | filename: script.py
# | code-line-numbers: true

import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import keras
from comet_ml import Experiment
import os


def evaluate(train_model_path, test_path, output_path, experiment):
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)
    
    # Load training model
    with tarfile.open(Path(train_model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(train_model_path))
    train_model = keras.models.load_model(Path(train_model_path) / "001")
    
    # Evaluate model
    train_predictions = np.argmax(train_model.predict(X_test), axis=-1)
    
    # Overall metrics
    train_accuracy = accuracy_score(y_test, train_predictions)
    train_precision = precision_score(y_test, train_predictions, average='weighted')
    train_recall = recall_score(y_test, train_predictions, average='weighted')
    
    print(f"Training model - Overall test accuracy: {train_accuracy}")
    print(f"Training model - Overall test precision: {train_precision}")
    print(f"Training model - Overall test recall: {train_recall}")
    
    # Initialize evaluation report with new structure
    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": train_accuracy,
                "description": "Overall model accuracy"
            },
            "precision": {
                "value": train_precision,
                "description": "Overall model precision"
            },
            "recall": {
                "value": train_recall,
                "description": "Overall model recall"
            }
        },
        "class_metrics": {}
    }
    
    # Per-class metrics
    class_names = ['Admit', 'Deny', 'Waitlist']
    train_cm = confusion_matrix(y_test, train_predictions)
    
    print("Training model - Confusion Matrix:\n", train_cm)
    
    # Calculate and store per-class metrics with descriptions
    for i, class_name in enumerate(class_names):
        # Calculate metrics
        train_class_accuracy = train_cm[i, i] / train_cm[i].sum() if train_cm[i].sum() > 0 else 0
        train_class_precision = precision_score(y_test == i, train_predictions == i, zero_division=0)
        train_class_recall = recall_score(y_test == i, train_predictions == i, zero_division=0)
        
        # Store metrics with descriptions
        evaluation_report["class_metrics"][class_name] = {
            "accuracy": {
                "value": train_class_accuracy,
                "description": f"Accuracy for {class_name} class"
            },
            "precision": {
                "value": train_class_precision,
                "description": f"Precision for {class_name} class"
            },
            "recall": {
                "value": train_class_recall,
                "description": f"Recall for {class_name} class"
            }
        }
        
        print(f"\nTraining model - {class_name} metrics:")
        print(f"Accuracy: {train_class_accuracy}")
        print(f"Precision: {train_class_precision}")
        print(f"Recall: {train_class_recall}")

    print("Saving evaluation report to designated path")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "train_model_report.json", "w") as f:
        f.write(json.dumps(evaluation_report, indent=4))
    
    # Log metrics to Comet.ml if experiment is set up
    if experiment:
        experiment.log_metrics({
            "training_accuracy": train_accuracy,
            "training_precision": train_precision,
            "training_recall": train_recall
        })
        for class_name, metrics in evaluation_report["class_metrics"].items():
            experiment.log_metrics({
                f"training_{class_name}_accuracy": metrics["accuracy"]["value"],
                f"training_{class_name}_precision": metrics["precision"]["value"],
                f"training_{class_name}_recall": metrics["recall"]["value"],
            })
        experiment.log_confusion_matrix(
            y_test.astype(int), train_predictions.astype(int),
            title="Training Model Confusion Matrix"
        )


if __name__ == "__main__":
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
    
    if experiment:
        experiment.set_name('Evaluating Training Model')
    
    evaluate(
        train_model_path="/opt/ml/processing/model",
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/",
        experiment=experiment
    )
