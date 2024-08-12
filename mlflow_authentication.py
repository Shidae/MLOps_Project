from mlflow.server import get_app_client
import os

tracking_uri = "http://127.0.0.1:5000/"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)

# Set username and password
os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_password"

# MLflow server is running, but specific end points update_user_password is not exist. MLflow's REST API does not have a built-in end point for updating user password. It does not support user management.


# Change password
auth_client.update_user_password(username="your_username", password="your_password")