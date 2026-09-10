import streamlit as st
import pandas as pd
import joblib


# Load the trained Random Forest model
rf_model = joblib.load("paysim_fraud_detection_model.pki")


# App title
st.title("PaySim Fraud Detection")

st.write(
    "Enter the transaction details below to predict whether the transaction "
    "is fraudulent."
)


# Transaction inputs
amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=0.0
)

oldbalanceOrg = st.number_input(
    "Original Account Balance Before Transaction",
    min_value=0.0,
    value=0.0
)

newbalanceOrig = st.number_input(
    "Original Account Balance After Transaction",
    min_value=0.0,
    value=0.0
)

oldbalanceDest = st.number_input(
    "Destination Account Balance Before Transaction",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "Destination Account Balance After Transaction",
    min_value=0.0,
    value=0.0
)


# Predict button
if st.button("Predict Fraud"):

    # Calculate the same fraud signals used during model training

    signal_1 = int(amount == oldbalanceOrg)

    signal_2 = int(amount > oldbalanceOrg)

    signal_3 = int(
        (amount == oldbalanceOrg)
        and (oldbalanceDest == 0)
        and (newbalanceDest == amount)
    )

    signal_4 = int(
        (amount == oldbalanceOrg)
        and (oldbalanceDest > newbalanceDest)
    )

    signal_5 = int(
        (amount == oldbalanceOrg)
        and (oldbalanceDest == 0)
        and (newbalanceDest == 0)
    )

    signal_6 = int(
        (amount == oldbalanceOrg)
        and (amount > (newbalanceDest - oldbalanceDest))
    )


    # Create the transaction using the exact feature order
    input_data = pd.DataFrame([[
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        signal_1,
        signal_2,
        signal_3,
        signal_4,
        signal_5,
        signal_6
    ]], columns=[
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_5",
        "signal_6"
    ])


    # Make prediction
    prediction = rf_model.predict(input_data)

    # Get fraud probability
    probability = rf_model.predict_proba(input_data)[0][1]


    # Display result
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction")

    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"**Fraud Probability: {probability:.2%}**")