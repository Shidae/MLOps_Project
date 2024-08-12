# text classification task using DistilBert model

import os
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, get_dataset_config_names
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW

# define parameters as a dict, iterate through them when logging



params = {
    'model_name': 'distilbert-base-cased',
    'learning_rate': 5e-5,
    'batch_size': 16,
    'num_epochs': 1,
    'dataset_name': 'ag_news', # default
    'task_name': 'sequence_classification',
    'log_steps': 100,
    'max_seq_length': 128,
    'output_dir': 'models/distilbert-base-uncased-aq-news'
}
"""
dataset_names = params['dataset_name']
configs = get_dataset_config_names(dataset_names)
print(configs)
"""
# Mlflow setup to log all metrics, params, and artifacts to the mlflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Runs are tracked in the experiment with the task name interpolated from the list
mlflow.set_experiment(f"{params['task_name']}")

# Use NLP dataset to ensure reproducibility and comparability.
# Convert raw text into a format required by our model.

# load from Huggingface dataset library, name and sub-name of the data
dataset = load_dataset(params['dataset_name']) #  ,params['task_name']

# Initialize tokenizer model, convert text data into format readable by DistilBert model
tokenizer = DistilBertTokenizer.from_pretrained(params['model_name'])

# tokenize batch of text data
# padding extends longer sequences to a specified length by adding [PAD] tokens for efficient batch processing. 
def tokenize(batch): # dict type
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=params['max_seq_length'])

# access training split, shuffles the data, select the first 20,000 examples from the shuffled set, apply (map) the tokenize function to each batch of the dataset
train_dataset = dataset["train"].shuffle().select(range(20_000)).map(tokenize, batched=True)
test_dataset = dataset["test"].shuffle().select(range(2_000)).map(tokenize, batched=True)

# Set format (to tokenize) for PyTorch and create data loaders
# convert to torch (PyTorch tensors) format required by the model
# specify columns that should be converted to PyTorch tensor like token ids of the input text, masks indicating which tokens are actual and which are padded, and classification label 
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# DataLoader groups these tokenized sequences into batches of 16, each batch has 16 text sequences (samples), each has 128 tokens due to tokenization.
# size 20000/16 = 1250 total batches
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

# get unique labels from the dataset
labels = dataset["train"].features['label'].names

# Initialize model, load pre trained model and ensure output layer of the model matches the number of classes
model = DistilBertForSequenceClassification.from_pretrained(params['model_name'], num_labels=len(labels))

# this dict maps class indices to corresp human readable label names(id to correp label)
model.config.id2label = {i: label for i , label in enumerate(labels)}
params['id2label'] = model.config.id2label # e.g. Output: {0: 'negative', 1: 'positive'}

# check if gpu(cuda) is available, so computations (forward and backward passes) perform on gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)

# Choosing optimizer
optimizer = AdamW(model.parameters(), lr=params['learning_rate'])

# Evaluate model on test set
def evaluate_model(model, dataloader, device):
    model.eval() # set model to evaluation mode, disable training activities like dropout etc.
    predictions, true_labels = [], []

    with torch.no_grad(): # disable gradient calculation when making predictions
        for batch in dataloader:
            # move input IDs, masks and labels to CPU/GPU
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

            # Perform forward pass, to get the raw output scores (logit prediction)
            # tokenized input as id and masks are given to model
            outputs = model(inputs, attention_mask = masks)
            # model produces multiple outputs (logits) to represent confidence scores for each possible class.
            logits = outputs.logits

            # Find index of the maximum logit value (highest confidence) along dimension 1, corresp to the predicted class label
            _, predicted_labels = torch.max(logits, dim=1)

            # Convert predicted and true labels to NumPy and stores in lists
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate Evaluation Metrics
    # macro calculate metrics independently and takes the average.
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

    return accuracy, precision, recall, f1

# Training loop, log metrics and parameters at each step

# Start MLflow Run
with mlflow.start_run(run_name=f"{params['model_name']}-{params['dataset_name']}") as run:

    # Log all params declared in the beginning at once
    mlflow.log_params(params)

    with tqdm(total=params['num_epochs'] * len(train_loader), desc=f"Epoch [1/{params['num_epochs']}] - (Loss: N/A) - Steps") as pbar:
        for epoch in range(params['num_epochs']):
            running_loss = 0.0
            for i, batch in enumerate(train_loader, 0): # e.g. start=1 index: 1, 2, 3
                inputs, masks, labels = batch['input_ids'].to(device), batch ['attention_mask'].to(device), batch['label'].to(device) # shape [16, 128], [16, 128], [16]

            optimizer.zero_grad() # clears old gradient
            outputs = model(inputs, attention_mask=masks, labels=labels) # forward pass to make prediction
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i and i % params['log_steps'] == 0:
                avg_loss = running_loss / params['log_steps']

                pbar.set_description(f"Epoch [{epoch + 1}/{params['num_epochs']}] - (Loss: {avg_loss:.3f}) - Steps")
                mlflow.log_metric("loss", avg_loss, step=epoch * len(train_loader) + i)

                running_loss = 0.0

            pbar.update(1)

        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1} Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} ")

        # Log metrics to MLflow
        mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, step=epoch)

    # Log model to MLflow through built-in PyTorch method
    mlflow.pytorch.log_model(pytorch_model=model, artifact_path= "model")

    # Log model to MLflow through custom method (not recommended because missing meta-data)
    #os.makedirs(params['output_dir'], exist_ok=True)
    #model.save_pretrained(params['output_dir'])
    #tokenizer.save_pretrained(params['output_dir'])

    #mlflow.log_artifacts(params['output_dir'], artifact_path="model")

    # Register the trained model in the MLflow model registry to refer later
    # specify the location of the model artifact
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "agnews-transformer")

print('Finished Training')

