
# Project Title

This project is designed to showcase various MLOps techniques including model deployment, quantization, batching, and dynamic batching in a machine learning workflow. The repository contains scripts and resources to demonstrate the implementation and optimization of machine learning models using tools like MLflow and CUDA.



## Installation

To set up the project locally, follow these steps:

Clone the repository:

```bash 
  git clone https://github.com/Shidae/MLOps_Project.git
  cd MLOps_Project

```
Create a virtual enrvironment:
```bash
  python -m venv venv
  source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
Install the required dependencies:
```bash
  pip install -r requirements_local.txt

```
Install additional dependencies for CUDA and MLflow if needed:

Follow the instructions in ```cuda_colab_install.py``` to set up CUDA in a Google Colab environment.

    
## Usage

This project includes multiple Python scripts for different stages of model deployment and optimization:

- **Training**: The `training_loop.py` script handles model training.
- **Model Deployment**:
  - `first_deployment.py`: Initial deployment script.
  - `second_deployment.py`: Deployment with enhanced batching.
  - `third_deployment_quantization.py`: Deployment with model quantization.
  - `fourth_deployment_paged_attention.py`: Deployment with paged attention mechanism.
- **Batching**:
  - `batching_and_dynamic_batching.py`: Script for batching and dynamic batching.
  - `sorting_batches.py`: Script for sorting batches.
- **MLflow**:
  - `mlflow_authentication.py`: Handles MLflow authentication.
  - `mlflow_inference.py`: Script for running inference using MLflow.

## Contributing

Contributions to this project are welcome. If you have any suggestions, feel free to create an issue or submit a pull request.
