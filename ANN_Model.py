
# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load Dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# Display first five rows
print(dataset.head())

# Check dataset shape
print("Dataset Shape:", dataset.shape)

# Check dataset information
print(dataset.info())



# Separate Independent and Dependent Variables

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print("Independent Variables (X):")
print(X[:5])

print("\nDependent Variable (y):")
print(y[:5])

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# Label Encoding Gender
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One-Hot Encoding Geography
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [1])
    ],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

print("Encoded X:")
print(X[:5])

print("Encoded X Shape:", X.shape)


# Splitting the dataset into Training and Testing set

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)


# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Scaled X_train:")
print(X_train[:5])

print("\nScaled X_test:")
print(X_test[:5])

# Initialize the ANN

ann = Sequential()

# Adding the input layer and first hidden layer

ann.add(Dense(units=6, activation='relu', input_dim=12))

# Adding the second hidden layer

ann.add(Dense(units=6, activation='relu'))

# Adding the output layer

ann.add(Dense(units=1, activation='sigmoid'))

ann.summary()

# Compiling the ANN

ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training the ANN

ann.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100
)

# Predicting the Test set results

y_pred = ann.predict(X_test)

print("Predicted Probabilities:")
print(y_pred[:10])

# Converting probabilities into binary predictions

y_pred = (y_pred > 0.5)

print("Predicted Values:")
print(y_pred[:10])

print("\nActual Values:")
print(y_test[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy * 100, "%")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Change prediction threshold

y_pred = (ann.predict(X_test) > 0.4).astype(int)

# Evaluate again

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Test Accuracy:", accuracy_score(y_test, y_pred) * 100)

# Plotting Confusion Matrix

plt.figure(figsize=(6, 5))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('ANN Confusion Matrix - Threshold 0.4')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

plt.show()

# Generate predictions
y_pred = (ann.predict(X_test) > 0.4).astype(int)

# Calculate confusion matrix once
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred) * 100)

# Plot the same confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('ANN Confusion Matrix - Threshold 0.4')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# Saving the ANN model

ann.save("customer_churn_model.keras")

print("ANN Model saved successfully!")

# Saving preprocessing objects

import joblib

joblib.dump(le, "gender_encoder.pkl")
joblib.dump(ct, "preprocessor.pkl")
joblib.dump(sc, "scaler.pkl")

print("Preprocessing objects saved successfully!")