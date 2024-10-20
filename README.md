Essay Scorer Project
Overview
This project implements an automated essay scoring system using BERT embeddings and machine learning models such as XGBoost, Linear Regression, Random Forest. The system processes essays, generates embeddings using a BERT model, and predicts essay scores using trained regression models. The project also evaluates the model's performance using metrics such as Root Mean Squared Error (RMSE) and Quadratic Weighted Kappa (QWK).

Project Structure
data/: Contains the training dataset (train.csv) and any other datasets you may need such as test or validation data.
app/: Includes the preprocessing scripts (e.g., preprocess.py) for cleaning essays.
    : Contains core scripts for training, evaluation, and inference.
    : Trained BERT embeddings are saved as .pkl files.
notebooks/: Contains any Jupyter notebooks used for EDA & experimentation.

Requirements
To run the project, you'll need to install the following libraries:

pip install torch transformers scikit-learn xgboost joblib pandas numpy tqdm


Make sure that you have the BERT model and tokenizer downloaded using Hugging Face’s transformers library, and that you have access to a GPU for faster training.

Models
This project uses the following models:

XGBoost Regressor: Tuned using RandomizedSearchCV to predict essay scores.
Linear Regression: For baseline predictions.
Random Forest Regressor (optional): For comparison against XGBoost and Linear Regression.

Files
train_rev1.py: Main script to train and save the model using BERT embeddings.
evaluate_model.py: Script to evaluate the model on a test dataset.
preprocess.py: Script to clean and preprocess essays.
model.pkl, tokenizer.pkl, bert_model.pkl: Saved models, tokenizer, and BERT model used for inference.

Training the Model
Preprocess the Data:

The script preprocess.py processes the essays by cleaning the text and preparing it for BERT embeddings.
Generate BERT Embeddings:

train_rev1.py generates BERT embeddings for each essay and splits the data into training and test sets.
Train the Models:

The models are trained using RandomizedSearchCV with predefined hyperparameter grids. The script uses XGBoost and Linear Regression models.
The training process tracks progress using tqdm and saves the best model based on performance on the validation set.
Hyperparameter Tuning
RandomizedSearchCV is used to tune the hyperparameters of both models. Hyperparameters are defined in the training script, and the search is limited to ensure the process completes in a reasonable amount of time.

XGBoost Hyperparameters:
n_estimators: [50, 100, 200]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.1, 0.2]
Linear Regression Hyperparameters:
fit_intercept: [True, False]
Evaluation Metrics
Root Mean Squared Error (RMSE): Measures the average difference between predicted and actual scores.
Quadratic Weighted Kappa (QWK): Measures the agreement between predicted and actual scores, adjusted for chance.
Inference
The model can be used to predict essay scores on a new dataset using the evaluate_model.py script. The script generates BERT embeddings for new essays and outputs the predicted scores along with the actual scores.


Example Usage
Train the Model:

python train_model.py

Evaluate the Model:

python evaluate_model.py --test_csv_path path/to/test_data.csv


Note: App is dockerized and fastapi will run on localhost and can test during interview.
