import mlflow

# MLflow manage ML lifecycle, include tools for tracking experiements, reproducible runs and sharing and deploying models.
# Set up tracking server, where all experiment data will be tracked and stored here. Runs at port 5000, remember disable firewall to this port
mlflow.set_tracking_uri("http://localhost:5000")

# Create new experiment, each contains multiple runs
experiment_name = "My new experiment97"
# Check if experiment exist
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    # if exist, get id
    experiment_id = experiment.experiment_id

else:
    experiment_id = mlflow.create_experiment(experiment_name)

# Start a new run and auto end run
with mlflow.start_run(experiment_id=experiment_id):
    # your ML code goes here
    pass

# Manually creating a custom named run
run = mlflow.start_run(experiment_id=experiment_id, run_name="First run")

# Logging parameters, keep track which settings used in each run for comparison and reproducibility
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("batch_size", 32)
num_epochs = 10
mlflow.log_param("num_epochs", num_epochs)


import numpy as np

# Logging metrics for each epoch to measure performance of model
# epoch: a single pass through the entire dataset or
# step: a pass of a batch of data
for epoch in range(num_epochs):
    mlflow.log_metric("accuracy", np.random.random(), step=epoch)
    mlflow.log_metric("loss", np.random.random(), step= epoch)

# Logging a custom time-series metric
for t in range(100):
    metric_value = np.sin(t * np.pi / 50)
    mlflow.log_metric("time_series_metric", metric_value, step=t)

# Logging datasets
with open("data/dataset.csv", "w") as f:
    f.write("x,y\n") # write the first roq with column headers x,y
    for x in range(100):
        f.write(f"{x}, {x * 2}\n") # fill columns

mlflow.log_artifact("data/dataset.csv", "data")


import pandas as pd
import plotly.express as px

# Generate a confusion matrix
confusion_matrix = np.random.randint(0, 100, size=(5, 5))  # 5x5 matrix

labels = ["Class A", "Class B", "Class C", "Class D", "Class E"]
df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

# Plot confusion matrix using Plotly Express
fig = px.imshow(df_cm, text_auto=True, labels=dict(x="Predicted Label", y="True Label"), x=labels, y=labels, title="Confusion Matrix")

# Save the figure as an HTML file
html_file = "confusion_matrix.html"
fig.write_html(html_file)

# Log the HTML file with MLflow
mlflow.log_artifact(html_file)

from transformers import AutoModelForSeq2SeqLM

# Initialize a model from Hugging Face Transformers
model = AutoModelForSeq2SeqLM.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")


# Log the model in MLflow
mlflow.pytorch.log_model(model, "transformer_model")


# logging metric
mlflow.end_run()