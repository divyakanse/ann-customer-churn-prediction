# 🏦 Bank Customer Churn Prediction using Artificial Neural Network

**AI-powered customer churn prediction using Artificial Neural Networks and Streamlit.**

🚀 **Live Demo:** [Try the Application](https://ann-customer-churn-prediction-nn9tw4uaym5aoxwzhbiuit.streamlit.app/)

💻 **GitHub Repository:** [View Source Code](https://github.com/divyakanse/ann-customer-churn-prediction)

---

## 📌 Project Overview

This project predicts whether a bank customer is likely to leave the bank using an Artificial Neural Network (ANN).

The application allows users to enter customer information and receive a predicted churn probability through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze customer information for churn prediction.
- Build and train an Artificial Neural Network.
- Apply data preprocessing and feature scaling.
- Predict customer churn probability.
- Develop an interactive web application using Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib and Seaborn
- Streamlit
- Joblib

## 🔄 Machine Learning Workflow

1. Load and explore the customer churn dataset.
2. Separate independent and dependent variables.
3. Encode categorical features.
4. Split data into training and testing sets.
5. Apply feature scaling.
6. Build an ANN using Keras Sequential.
7. Train and evaluate the model.
8. Save the trained model and preprocessing objects.
9. Build a Streamlit application for customer predictions.

## 🧠 ANN Architecture

- Input layer: 12 features
- Hidden layer 1: 6 neurons, ReLU activation
- Hidden layer 2: 6 neurons, ReLU activation
- Output layer: 1 neuron, Sigmoid activation
- Optimizer: Adam
- Loss function: Binary Crossentropy

## 📊 Application Features

- Customer information input form
- Geography and gender selection
- ANN-based churn probability prediction
- Customer risk classification
- Suggested customer retention action
- Interactive Streamlit dashboard

## ⚙️ Decision Threshold

The application uses a 0.40 probability threshold to classify customers.

- Probability above 0.40: Likely to leave
- Probability at or below 0.40: Likely to stay

This threshold is an experimental modelling choice and can be adjusted according to business requirements.

## 📁 Project Structure

```text
ANN Model/
├── ANN_Model.py
├── app.py
├── Churn_Modelling.csv
├── customer_churn_model.keras
├── gender_encoder.pkl
├── preprocessor.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/divyakanse/ann-customer-churn-prediction.git
```

### 2. Open the project folder

```bash
cd ann-customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 🔮 Future Improvements

- Add model performance visualizations.
- Improve customer risk segmentation.
- Add explainable AI techniques.
- Improve the dashboard with additional business insights.

## ⚠️ Disclaimer

This project is developed for educational and portfolio purposes. Predictions are estimates and should not be treated as guarantees of customer behaviour.
