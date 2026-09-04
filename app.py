
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    "model_fixed.h5",
    compile=False
)


# ============================================================
# LOAD ENCODERS AND SCALER
# ============================================================

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("Customer Churn Prediction")

st.write(
    "Enter customer details to predict churn probability."
)


# ============================================================
# USER INPUT
# ============================================================

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=30
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Churn"):

    # --------------------------------------------------------
    # Encode Gender
    # --------------------------------------------------------

    gender_encoded = label_encoder_gender.transform(
        [gender]
    )[0]


    # --------------------------------------------------------
    # Create input DataFrame
    # IMPORTANT:
    # These names must match the names used while fitting
    # scaler.pkl
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })


    # --------------------------------------------------------
    # One-Hot Encode Geography
    # --------------------------------------------------------

    geo_encoded = onehot_encoder_geo.transform(
        [[geography]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out()
    )


    # --------------------------------------------------------
    # Combine numerical/categorical data + geography
    # --------------------------------------------------------

    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geo_encoded_df.reset_index(drop=True)
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Make sure columns are in the SAME ORDER as scaler
    # --------------------------------------------------------

    if hasattr(scaler, "feature_names_in_"):

        expected_columns = list(
            scaler.feature_names_in_
        )

        # Check if any required column is missing
        missing_columns = [
            col
            for col in expected_columns
            if col not in input_data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns required by scaler: "
                f"{missing_columns}"
            )
            st.stop()

        # Reorder columns exactly like training
        input_data = input_data[
            expected_columns
        ]


    # --------------------------------------------------------
    # Scale input data
    # --------------------------------------------------------

    input_data_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_probability = float(
        prediction[0][0]
    )


    # --------------------------------------------------------
    # Display Result
    # --------------------------------------------------------

    st.subheader("Prediction Result")


    if prediction_probability > 0.5:

        st.error(
            "The customer is likely to churn."
        )

    else:

        st.success(
            "The customer is unlikely to churn."
        )


    # --------------------------------------------------------
    # Display Probability
    # --------------------------------------------------------

    st.write(
        f"Churn Probability: "
        f"{prediction_probability:.2%}"
    )
