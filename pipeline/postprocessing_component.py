#| filename: postprocessing_component.py
#| code-line-numbers: true

import os
import numpy as np
import json
import joblib


try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    # We don't have access to the `worker` package when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None


def model_fn(model_dir):
    """
    Deserializes the target model and returns the list of fitted categories.
    """

    model = joblib.load(os.path.join(model_dir, "target.joblib"))
    #Returns the categories (admission statuses) that the model was trained on
    return model.named_transformers_["admission"].categories_[0]

#input data from after prediction, the input data contains the answer and we get the predictions
def input_fn(input_data, content_type):
    if content_type == "application/json":
        return json.loads(input_data)["predictions"]
    
    raise ValueError(f"{content_type} is not supported.")


def predict_fn(input_data, model):
    """
    Transforms the prediction into its corresponding category.
    """
    #predictions will change to 0,1,2, depending on location of highest value
    predictions = np.argmax(input_data, axis=-1)
    #give the max value of the input data, give the probability, 0.001,0.9,0.009
    confidence = np.max(input_data, axis=-1)
    #model is going to convert 0,1,2 to the admission status
    return [
        (model[prediction], confidence)
        for confidence, prediction in zip(confidence, predictions)
    ]

#kinda answer the receptor wants
def output_fn(prediction, accept):
    if accept == "text/csv":
        return (
            worker.Response(encoders.encode(prediction, accept), mimetype=accept)
            if worker
            else (prediction, accept)
        )

    #give json kinda answer
    if accept == "application/json":
        response = []
        for p, c in prediction:
            response.append({"prediction": p, "confidence": c})

        # If there's only one prediction, we'll return it
        # as a single object.
        if len(response) == 1:
            response = response[0]

        return (
            worker.Response(json.dumps(response), mimetype=accept)
            if worker
            else (response, accept)
        )

    raise Exception(f"{accept} accept type is not supported.")
