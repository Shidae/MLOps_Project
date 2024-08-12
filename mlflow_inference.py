# MLflow Integration of Model Serving and Registry Management in the operational phase of ML lifecycle.
# serving models for inference and managing multiple versions of the model, transition through stages in the model lifecycle, delete models and versions

# %%
import mlflow
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


# connect to mlflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

# Two ways to retrieve model from MLflow

# 1st Method: Using Built-in PyTorch Loader
model_name = "agnews-transformer"
model_version = "2" # or production or staging

# Constructs a reference to a specific version of a registered model in the registry
model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.pytorch.load_model(model_uri)

# %%

# predict function to make inference using the loaded model.
# takes a list of tests, and tokenizes them using a pre-trained tokenizer, and used by the model.

def predict(texts, model, tokenizer):
    # Tokenize the text
    # to make predictions, inputs must be in the same device as the model -> GPU
    # this return as a list of python integers, pt converts into pytorch tensors
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(model.device)

    # Pass the inputs to the model
    with torch.no_grad():
        # unpack the input dict into inputID and attention mask
        outputs = model(**inputs) # [0, 1, 2, 3] for four categories

        # predicted class indices [1, 1] -> ["Sports", "Sports"]
        predictions = torch.argmax(outputs.logits, dim=-1) # find argument across column, not rows.

    # Convert predictions to text labels
    # numpy cannot operate directly on GPU tensors.
    predictions = predictions.cpu().numpy()

    # find the label to the corresponding index (id)
    predictions = [model.config.id2label[prediction] for prediction in predictions]

    return predictions

# Sample text to predict
texts = [
    "The local high school soccer team triumphed in the state championship, securing victory with a last-second winning goal.",
    "DataCore is set to acquire startup InnovateAI for $2 billion, aiming to enhance its position in the artificial intelligence market.",
]


# Tokenizer needs to be loaded separately for this
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print(predict(texts, model, tokenizer=tokenizer))

# %%
# 2nd Method: Versatile Loading with Custom Handling
# Useful when handling variety of models

"""
# Load custom model
model_name = "agnews-transformer"
model_version = "2"
model_version_details = client.get_model_version(name=model_name, version=model_version)

# Access model files (artifacts) and with specific run
run_id = model_version_details.run_id
artifact_path = model_version_details.source

# Construct the model URI (location of the model, uniform resource identifier)
# reference a model in the mlflow model registry
model_uri = f"models:/{model_name}/{model_version}"

# create directory for model artifacts
model_path = "models/agnews_transformer"
os.makedirs(model_path, exist_ok=True) # prevents error if dir already exist

# Download model files (artifacts) associated with specific run from the tracking server to local directory
client.download_artifacts(run_id, artifact_path, dst_path=model_path)


# Load the model and tokenizer
custom_model = AutoModelForSequenceClassification.from_pretrained("model/agnews/transformer/custom_model")
tokenizer = AutoTokenizer.from_pretrained("models/agnews_transformer/custom_model")

# Inference
print(predict(texts, custom_model, tokenizer=tokenizer))


"""

# Model Versioning to handle multiple versions of models.
# Set a new experiment and logging models under differenr run names, can create multiple versions of the same model.
# can track evolution of models over time.
# e.g. log two additional iterations as "iteration2" and "iteration3"

# Log new models for versioning demonstration
# if it does not exist, mlflow creates it, contains runs
mlflow.set_experiment("sequence_classification")

# Log a new model and start run as iteration 2
with mlflow.start_run(run_name="iteration2"):
    mlflow.pytorch.log_model(model, "model")

# Log another new model as iteration 3
with mlflow.start_run(run_name="iteration3"):
    mlflow.pytorch.log_model(model, "model")

# %%

# Model version management
# Access all model versions
model_versions = client.search_model_versions(f"name='{model_name}'")
for version in model_versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}")

# Change model stage
client.transition_model_version_stage(name=model_name, version=model_version, stage="Production")

# Deleting models and versions
# delete specific or entire registered models from the registry.

# Delete a specific model version
# client.delete_model_version(name=model_name, version=model_version)

# Delete the entire registered model
# client.delete_registered_model(name=model_name)
# %%
