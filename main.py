# main.py
# This file contains the FastAPI application to serve the trained Iris classification model.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- 1. Model Training and Saving (This part would typically be run once offline) ---
# For the purpose of this assignment, we'll include it here to make the code self-contained.

MODEL_PATH = "iris_logistic_regression_model.pkl"
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

def train_and_save_model():
    """
    Trains a Logistic Regression model on the Iris dataset and saves it to a file.
    This function simulates the data scientist's role in creating and saving the model.
    """
    print("--- Training and Saving Model ---")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into training and testing sets (optional for this simple demo, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Logistic Regression model
    # Logistic Regression is a linear model used for binary classification.
    # Despite its name, it can be extended to multi-class classification (like Iris)
    # using strategies like One-vs-Rest (OvR) or Multinomial Logistic Regression.
    # It estimates the probability of an instance belonging to a particular class.
    # The 'solver' parameter specifies the algorithm to use for optimization.
    # 'lbfgs' is a good default for multi-class problems.
    # 'max_iter' is increased to ensure convergence.
    model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model (for demonstration)
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")

    # Save the trained model to a file using joblib
    # joblib is efficient for large NumPy arrays and is often preferred over pickle
    # for scikit-learn models.
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("-" * 30)

# Ensure the model is trained and saved before the FastAPI app starts
if not os.path.exists(MODEL_PATH):
    train_and_save_model()
else:
    print(f"Model already exists at {MODEL_PATH}. Skipping training.")
    print("-" * 30)


# --- 2. FastAPI Application Setup ---

# Initialize the FastAPI application
app = FastAPI(
    title="Iris Species Prediction API",
    description="A simple API to predict Iris flower species (setosa, versicolor, virginica) "
                "based on sepal and petal measurements using a pre-trained Logistic Regression model.",
    version="1.0.0"
)

# Load the pre-trained model when the FastAPI application starts up.
# This ensures the model is loaded only once, not for every incoming request,
# which is crucial for performance.
try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real-world scenario, you might want to raise an exception and prevent
    # the application from starting if the model cannot be loaded.
    model = None # Set model to None to indicate failure

# --- 3. Pydantic Models for Request and Response Data ---

# Pydantic `BaseModel` is used to define the structure and data types of your API's inputs.
# FastAPI uses these models for automatic data validation, serialization, and documentation.
class IrisFeatures(BaseModel):
    """
    Defines the expected input features for the Iris prediction endpoint.
    Each field is defined with its type and a description for documentation.
    `Field` can be used to add more validation (e.g., min/max values) and examples.
    """
    sepal_length: float = Field(..., example=5.1, description="Length of the sepal in cm")
    sepal_width: float = Field(..., example=3.5, description="Width of the sepal in cm")
    petal_length: float = Field(..., example=1.4, description="Length of the petal in cm")
    petal_width: float = Field(..., example=0.2, description="Width of the petal in cm")

    class Config:
        # This inner class provides configurations for Pydantic models.
        # `schema_extra` can be used to provide example values that appear in the
        # auto-generated OpenAPI (Swagger UI) documentation.
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    """
    Defines the structure of the response returned by the prediction endpoint.
    """
    predicted_species: str = Field(..., example="setosa", description="The predicted Iris species")
    confidence: float = Field(..., example=0.98, description="Confidence score of the prediction")

# --- 4. API Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint for the API. Returns a simple welcome message.
    """
    return {"message": "Welcome to the Iris Species Prediction API! Visit /docs for API documentation."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_species(features: IrisFeatures):
    """
    Predicts the Iris species based on the provided sepal and petal measurements.

    **Input:**
    - `sepal_length`: Length of the sepal in cm (float)
    - `sepal_width`: Width of the sepal in cm (float)
    - `petal_length`: Length of the petal in cm (float)
    - `petal_width`: Width of the petal in cm (float)

    **Output:**
    - `predicted_species`: The predicted Iris species (e.g., "setosa", "versicolor", "virginica")
    - `confidence`: The confidence score of the prediction (float)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Machine learning model not loaded.")

    try:
        # Convert the Pydantic model instance into a NumPy array suitable for the model.
        # The order of features must match the order the model was trained on.
        input_data = np.array([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]).reshape(1, -1) # Reshape to (1, n_features) as expected by scikit-learn's predict method

        # Make a prediction using the loaded model
        prediction_index = model.predict(input_data)[0]
        predicted_species = CLASS_NAMES[prediction_index]

        # Get prediction probabilities (confidence scores)
        # model.predict_proba() returns probabilities for each class.
        # We take the probability of the predicted class.
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction_index]) # Convert to standard float for JSON serialization

        # Return the prediction and confidence
        return PredictionResponse(predicted_species=predicted_species, confidence=confidence)

    except Exception as e:
        # Implement robust error handling.
        # If the prediction fails for any reason, return a 500 Internal Server Error.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- 5. How to Run the Application ---

# To run this FastAPI application, you need to have `uvicorn` installed:
# pip install uvicorn fastapi scikit-learn joblib

# Then, run the following command in your terminal from the directory containing main.py:
# uvicorn main:app --reload

# `--reload` enables auto-reloading when code changes, useful for development.
# In production, you would typically run `uvicorn main:app` or use a process manager like Gunicorn:
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

# --- 6. How to Test the API (Sample Requests) ---

# Once the server is running (e.g., on http://127.0.0.1:8000), you can test it:

# A. Using the interactive Swagger UI:
# Open your browser and go to: http://127.0.0.1:8000/docs
# You can see the `/predict` endpoint, try it out, and send sample requests directly from the UI.

# B. Using curl from your terminal:

# Example 1: Iris-setosa like measurements
# Expected output: "setosa"
# curl -X POST "http://127.0.0.1:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "sepal_length": 5.1,
#            "sepal_width": 3.5,
#            "petal_length": 1.4,
#            "petal_width": 0.2
#          }'

# Example 2: Iris-versicolor like measurements
# Expected output: "versicolor"
# curl -X POST "http://127.0.0.1:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "sepal_length": 6.0,
#            "sepal_width": 2.7,
#            "petal_length": 4.2,
#            "petal_width": 1.3
#          }'

# Example 3: Iris-virginica like measurements
# Expected output: "virginica"
# curl -X POST "http://127.0.0.1:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "sepal_length": 6.3,
#            "sepal_width": 3.3,
#            "petal_length": 6.0,
#            "petal_width": 2.5
#          }'

# Example 4: Test root endpoint
# curl "http://127.0.0.1:8000/"

# --- 7. Mathematical Explanation of Logistic Regression ---

# Logistic Regression, despite its name, is a linear model for classification.
# It uses the logistic (sigmoid) function to map the output of a linear equation
# to a probability value between 0 and 1.

# For binary classification:
# The linear part: $z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$
# Where:
# - $z$ is the linear combination of features.
# - $w_0$ is the bias (intercept).
# - $w_i$ are the weights (coefficients) for each feature $x_i$.

# The logistic (sigmoid) function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
# This function squashes any real-valued number $z$ into a probability between 0 and 1.

# The probability of the positive class ($P(Y=1|X)$) is given by:
# $P(Y=1|X) = \sigma(z)$
# And the probability of the negative class ($P(Y=0|X)$) is $1 - \sigma(z)$.

# For multi-class classification (like Iris with 3 classes):
# Scikit-learn's LogisticRegression typically uses either:
# 1. One-vs-Rest (OvR) / One-vs-All (OvA):
#    - Trains a separate binary logistic regression model for each class.
#    - Each model predicts the probability of an instance belonging to its class versus all other classes.
#    - The class with the highest predicted probability across all models is chosen as the final prediction.
# 2. Multinomial Logistic Regression (Softmax Regression):
#    - Directly generalizes logistic regression to multi-class problems.
#    - It calculates a score for each class and then applies the softmax function to convert these scores into probabilities that sum to 1.
#    - The softmax function for class $k$ is:
#      $P(Y=k|X) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$
#      Where $z_k$ is the linear combination for class $k$, and $K$ is the total number of classes.

# The model learns the optimal weights ($w_i$) by minimizing a cost function (e.g., cross-entropy loss)
# using an optimization algorithm (like L-BFGS, which is used by default in scikit-learn for multi-class).

# --- 8. Deployment Considerations ---

# Dockerization:
# To deploy this application, you would typically containerize it using Docker.
# Create a `Dockerfile` in the same directory as `main.py`:

# Dockerfile content:
# FROM python:3.9-slim-buster
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# requirements.txt content:
# fastapi
# uvicorn
# scikit-learn
# joblib

# To build the Docker image:
# docker build -t iris-fastapi-app .

# To run the Docker container:
# docker run -p 8000:8000 iris-fastapi-app

# This makes your application portable and reproducible across different environments.

# Scaling:
# For production, you'd use a WSGI/ASGI server like Gunicorn to manage multiple Uvicorn worker processes.
# Example Gunicorn command (inside Docker or on a server):
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
# This runs 4 worker processes, allowing the application to handle more concurrent requests.
# For even higher scale, deploy on cloud platforms (AWS ECS/EKS, GCP Cloud Run/GKE, Azure AKS)
# that can automatically scale your Docker containers based on traffic.

# Monitoring:
# - Use logging within your FastAPI app (`import logging`) to track requests, errors, and model predictions.
# - Integrate with monitoring tools (e.g., Prometheus, Grafana) to track API latency, throughput, CPU/memory usage.
# - Implement model monitoring to detect data drift (changes in input data distribution) or concept drift (changes in the relationship between input and output).

# Error Handling:
# - FastAPI's `HTTPException` is used here to return standard HTTP error responses.
# - Custom exception handlers can be implemented for more specific error scenarios.
