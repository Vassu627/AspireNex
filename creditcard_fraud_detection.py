import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

data = pd.read_csv('creditcard.csv')

# Separate fraudulent and not fraudulent transactions
nofraud = data[data.Class == 0]
fraud = data[data.Class == 1]

# Balance both classes because they are imbalanced
nofraud_sample = nofraud.sample(n=len(fraud), random_state=2)
data = pd.concat([nofraud_sample, fraud], axis=0)

x = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

st.title("Credit Card Fraud Detection")
st.write("Enter Features to check if the transaction is legitimate or fraudulent")
st.info("Legitimate Transaction -> Non-Fraudulent Transaction (Normal Transaction)")

input_features = st.text_input("Enter Features")

submit = st.button("Detect")

if submit:
    input_values = input_features.split(',')
    
    # Validate input values are numeric
    try:
        values = np.array(input_values, dtype=np.float64)
    except ValueError:
        st.error("Invalid input! Please enter numeric values only.")
        st.stop()
    
    detection = model.predict(values.reshape(1, -1))

    if detection[0] == 0:
        st.subheader("Legitimate Transaction")
    else:
        st.subheader("Fraudulent Transaction")
