# 📈 Long-Term Investor Detection App

## 🔍 Overview
This Streamlit-based web application predicts whether an investor is a **Long-Term Investor** or **Short-Term Investor** based on their financial behavior. The app leverages **Machine Learning models** trained on a preprocessed dataset and provides interpretability using **SHAP**.

## 🏗️ Features
- User-friendly web interface for predictions
- Supports **Logistic Regression, Random Forest, and Decision Tree** models
- Handles categorical and numerical inputs efficiently
- Provides **SHAP Feature Importance** visualization
- Offers **LIME Explanations** for model predictions
- Tracks experiments using **MLflow**

## 🏁 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <your-repository-url>
cd long_term_investor_detection
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model
```bash
python train.py
```

### 5️⃣ Run the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

## 📊 Model Training
The training script (`train.py`) performs the following steps:
1. **Preprocesses Data**: Reads the `bank_marketing_cleaned.csv` dataset and applies preprocessing.
2. **Trains Models**: Trains **Logistic Regression, Random Forest, and Decision Tree** classifiers.
3. **Performs Hyperparameter Tuning**: Uses grid search for optimal hyperparameters.
4. **Logs Experiment Data**: MLflow is used to log model parameters, accuracy scores, and trained models.
5. **Saves Trained Models**: Saves models as `.pkl` files for deployment.

## 📌 Key Technologies
- **Python**
- **Streamlit** (for web app interface)
- **scikit-learn** (for ML models)
- **SHAP** (for model interpretability)
- **Pandas & NumPy** (for data handling)
- **MLflow** (for experiment tracking)
- **Joblib** (for model serialization)

## 🚀 Deployment Instructions
To deploy this application:
1. Ensure all dependencies are installed (`pip install -r requirements.txt`).
2. Train the model (`python train.py`).
3. Run the Streamlit app (`streamlit run streamlit_app/app.py`).
4. Access the app at `http://localhost:8501/`.

## 📢 Contributing
Feel free to fork the repository, create a new branch, and submit a pull request with your improvements!



