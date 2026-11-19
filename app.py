
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("customer_churn_model.keras")
    encoder = joblib.load("gender_encoder.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, encoder, preprocessor, scaler

ann, le, ct, sc = load_artifacts()

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #142B4A, #245B83);
    padding: 28px;
    border-radius: 14px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #DCE8F5;
    margin-bottom: 0;
}
div[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border: 1px solid #E1E7EF;
    border-radius: 12px;
}
.result-card {
    background: white;
    padding: 24px;
    border: 1px solid #E1E7EF;
    border-radius: 12px;
}
.small-note {
    color: #667085;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# Dashboard header
st.markdown("""
<div class="hero">
    <h1>🏦 Bank Customer Churn Analytics</h1>
    <p>AI-powered customer retention and churn prediction using Artificial Neural Networks</p>
</div>
""", unsafe_allow_html=True)

# Main layout
left_col, right_col = st.columns([1.2, 0.8], gap="large")

# Customer input form
with left_col:
    st.subheader("👤 Customer Information")
    st.caption("Enter customer details to estimate churn probability.")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.number_input(
                "Credit Score", 300, 900, 650
            )
            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"]
            )
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )
            age = st.number_input(
                "Age", 18, 100, 35
            )
            tenure = st.number_input(
                "Tenure (Years)", 0, 10, 5
            )

        with col2:
            balance = st.number_input(
                "Account Balance", min_value=0.0,
                value=50000.0, step=1000.0
            )
            num_products = st.selectbox(
                "Number of Products", [1, 2, 3, 4]
            )
            has_credit_card = st.selectbox(
                "Has Credit Card?", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            is_active_member = st.selectbox(
                "Active Member?", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            salary = st.number_input(
                "Estimated Salary", min_value=0.0,
                value=50000.0, step=1000.0
            )

        predict_button = st.form_submit_button(
            "🔍 Predict Customer Churn",
            use_container_width=True
        )

# Prediction section
with right_col:
    st.subheader("📊 Prediction Dashboard")

    if predict_button:

        input_data = np.array([[
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            num_products,
            has_credit_card,
            is_active_member,
            salary
        ]], dtype=object)

        # Encode gender
        input_data[0, 2] = le.transform([gender])[0]

        # Encode geography
        encoded_data = ct.transform(input_data)

        # Scale features
        scaled_data = sc.transform(encoded_data)

        # ANN prediction
        probability = float(
            ann.predict(scaled_data, verbose=0)[0][0]
        )

        threshold = 0.40

    
        st.caption("ESTIMATED CHURN PROBABILITY")
        st.metric(
            "Customer Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

        if probability > threshold:
            st.error("⚠️ HIGHER CHURN RISK: Likely to Leave")
            st.markdown("**Suggested action:** Review the customer's experience and consider suitable retention support.")
        else:
            st.success("✅ LOWER CHURN RISK: Likely to Stay")
            st.markdown("**Suggested action:** Continue engagement and maintain customer satisfaction.")

    
        st.caption(
            "Decision threshold: 40%. "
            "Predictions are estimates, not guarantees."
        )

    else:
        st.info(
            "👈 Enter customer information and click "
            "'Predict Customer Churn' to view the result."
        )

        st.markdown("""
        **How this application works**

        1. Collects customer information.
        2. Applies the saved preprocessing steps.
        3. Sends the processed data to the ANN model.
        4. Displays the estimated churn probability.
        """)

# Footer
st.divider()
st.caption(
    "Developed using Python | TensorFlow | Scikit-learn | Streamlit"
)
