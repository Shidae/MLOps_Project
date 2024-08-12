
# Project Title

This project, led by *The Fuzzy Scientist*, is designed to showcase various MLOps techniques including model deployment, quantization, batching, and dynamic batching in a machine learning workflow. The repository contains scripts and resources to demonstrate the implementation and optimization of machine learning models using tools like MLflow and CUDA.



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

- **Model Deployment**:
  - `training_loop.py`:  Shows text classification training pipeline using PyTorch and MLflow for experiment tracking.
  - `first_deployment.py`: Shows text generation using Hugging Face's transformers pipeline.
  - `second_deployment.py`: Illustrates manual model loading and text generation with Hugging Face's transformers.
  - `third_deployment_quantization.py`: Implements model quantization with ctranslate2 for efficient inference.
- **Batching**:
  - `batching_and_dynamic_batching.py`: Implements dynamic batching techniques for efficient text generation with PyTorch.
  - `sorting_batches.py`: Demonstrates text generation optimization using Hugging Face's transformers library.
  - `quantization.py`: Explores model quantization techniques using ctranslate2 for optimized deployment.
- **MLflow**:
  - `mlflow_authentication.py`: Handles MLflow authentication.
  - `mlflow_inference.py`: Demonstrates model serving and version management using MLflow.

## Contributing

Contributions to this project are welcome. If you have any suggestions, feel free to create an issue or submit a pull request.
