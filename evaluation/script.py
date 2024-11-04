# | filename: script.py
# | code-line-numbers: true


import json
import tarfile
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import keras
from comet_ml import Experiment


def evaluate(train_model_path, tune_model_path, test_path, output_path, experiment):
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)
    
    # Load both models
    with tarfile.open(Path(train_model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(train_model_path))
    train_model = keras.models.load_model(Path(train_model_path) / "001")
    
    with tarfile.open(Path(tune_model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(tune_model_path))
    tune_model = keras.models.load_model(Path(tune_model_path) / "001")
    
    # Evaluate both models
    train_predictions = np.argmax(train_model.predict(X_test), axis=-1)
    tune_predictions = np.argmax(tune_model.predict(X_test), axis=-1)
    
    # Overall metrics for both models
    train_accuracy = accuracy_score(y_test, train_predictions)
    train_precision = precision_score(y_test, train_predictions, average='weighted')
    train_recall = recall_score(y_test, train_predictions, average='weighted')
    
    tune_accuracy = accuracy_score(y_test, tune_predictions)
    tune_precision = precision_score(y_test, tune_predictions, average='weighted')
    tune_recall = recall_score(y_test, tune_predictions, average='weighted')
    
    print(f"Training model - Overall test accuracy: {train_accuracy}")
    print(f"Training model - Overall test precision: {train_precision}")
    print(f"Training model - Overall test recall: {train_recall}")
    
    print(f"Tuning model - Overall test accuracy: {tune_accuracy}")
    print(f"Tuning model - Overall test precision: {tune_precision}")
    print(f"Tuning model - Overall test recall: {tune_recall}")
    
    # Per-class metrics for both models, as defined in pre-processing
    class_names = ['Admit', 'Deny', 'Waitlist']
    train_cm = confusion_matrix(y_test, train_predictions)
    tune_cm = confusion_matrix(y_test, tune_predictions)
    
    print("Training model - Confusion Matrix:\n", train_cm)
    print("Tuning model - Confusion Matrix:\n", tune_cm)
    
    train_class_metrics = {}
    tune_class_metrics = {}
    for i, class_name in enumerate(class_names):
        # Training model metrics
        train_class_accuracy = train_cm[i, i] / train_cm[i].sum() if train_cm[i].sum() > 0 else 0
        train_class_precision = precision_score(y_test == i, train_predictions == i, zero_division=0)
        train_class_recall = recall_score(y_test == i, train_predictions == i, zero_division=0)
        
        train_class_metrics[class_name] = {
            "accuracy": train_class_accuracy,
            "precision": train_class_precision,
            "recall": train_class_recall
        }
        
        print(f"\nTraining model - {class_name} metrics:")
        print(f"Accuracy: {train_class_accuracy}")
        print(f"Precision: {train_class_precision}")
        print(f"Recall: {train_class_recall}")
        
        # Tuning model metrics
        tune_class_accuracy = tune_cm[i, i] / tune_cm[i].sum() if tune_cm[i].sum() > 0 else 0
        tune_class_precision = precision_score(y_test == i, tune_predictions == i, zero_division=0)
        tune_class_recall = recall_score(y_test == i, tune_predictions == i, zero_division=0)
        
        tune_class_metrics[class_name] = {
            "accuracy": tune_class_accuracy,
            "precision": tune_class_precision,
            "recall": tune_class_recall
        }
        
        print(f"\nTuning model - {class_name} metrics:")
        print(f"Accuracy: {tune_class_accuracy}")
        print(f"Precision: {tune_class_precision}")
        print(f"Recall: {tune_class_recall}")
    
    # Determine the best model
    best_model = "Training" if train_accuracy > tune_accuracy else "Tuning"
    best_accuracy = max(train_accuracy, tune_accuracy)
    best_model_path = train_model_path if best_model == "Training" else tune_model_path
    
    # Create an evaluation report
    evaluation_report = {
        "training_model": {
            "metrics": {
                "accuracy": {"value": train_accuracy},
                "precision": {"value": train_precision},
                "recall": {"value": train_recall},
            },
            "class_metrics": train_class_metrics,
            "path": train_model_path
        },
        "tuning_model": {
            "metrics": {
                "accuracy": {"value": tune_accuracy},
                "precision": {"value": tune_precision},
                "recall": {"value": tune_recall},
            },
            "class_metrics": tune_class_metrics,
            "path": tune_model_path
        },
        "best_model": {
            "type": best_model,
            "accuracy": best_accuracy,
            "path": best_model_path
        }
    }

    print("Saving evaluation report to designated path")


    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "report.json", "w") as f:
        f.write(json.dumps(evaluation_report, indent=4))

    
    # Log metrics to Comet.ml if experiment is set up
    if experiment:
        experiment.log_metrics({
            "training_accuracy": train_accuracy,
            "training_precision": train_precision,
            "training_recall": train_recall,
            "tuning_accuracy": tune_accuracy,
            "tuning_precision": tune_precision,
            "tuning_recall": tune_recall,
            "best_model_accuracy": best_accuracy
        })
        for class_name, metrics in train_class_metrics.items():
            experiment.log_metrics({
                f"training_{class_name}_accuracy": metrics["accuracy"],
                f"training_{class_name}_precision": metrics["precision"],
                f"training_{class_name}_recall": metrics["recall"],
            })
        for class_name, metrics in tune_class_metrics.items():
            experiment.log_metrics({
                f"tuning_{class_name}_accuracy": metrics["accuracy"],
                f"tuning_{class_name}_precision": metrics["precision"],
                f"tuning_{class_name}_recall": metrics["recall"],
            })
        experiment.log_confusion_matrix(
            y_test.astype(int), train_predictions.astype(int),
            title="Training Model Confusion Matrix"
        )
        experiment.log_confusion_matrix(
            y_test.astype(int), tune_predictions.astype(int),
            title="Tuning Model Confusion Matrix"
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
        experiment.set_name('Evaluating Training and Tuning Models')
    
    evaluate(
        train_model_path="/opt/ml/processing/train_model/",
        tune_model_path="/opt/ml/processing/tune_model/",
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/",
        experiment=experiment
    )
