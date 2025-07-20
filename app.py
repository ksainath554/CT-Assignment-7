import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource 
def train_dummy_model():
    """
    Generates synthetic data and trains a simple Logistic Regression model.
    """
    st.write("Training a dummy Logistic Regression model...")
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    df['Target'] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df[['Feature_1', 'Feature_2']], df['Target'], test_size=0.2, random_state=42
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Dummy model trained with accuracy: {accuracy:.2f}")
    return model, X, y 

model, X_data, y_data = train_dummy_model()


st.set_page_config(
    page_title="ML Model Deployment Demo",
    page_icon="🤖",
    layout="centered", 
    initial_sidebar_state="expanded" 
)

st.title("🤖 ML Model Deployment with Streamlit")
st.markdown("""
This application demonstrates how to deploy a machine learning model using Streamlit.
Input values for the features below and get a real-time prediction from our dummy classification model.
""")

st.sidebar.header("About the Model")
st.sidebar.info("""
This app uses a **Logistic Regression** model trained on synthetic data.
It predicts a binary outcome (0 or 1) based on two input features.
""")

st.header("Input Features")

feature_1 = st.slider(
    "Feature 1 Value",
    min_value=float(X_data[:, 0].min()),
    max_value=float(X_data[:, 0].max()),
    value=float(X_data[:, 0].mean()),
    step=0.01,
    help="Adjust this slider to change the value for Feature 1."
)

feature_2 = st.slider(
    "Feature 2 Value",
    min_value=float(X_data[:, 1].min()),
    max_value=float(X_data[:, 1].max()),
    value=float(X_data[:, 1].mean()),
    step=0.01,
    help="Adjust this slider to change the value for Feature 2."
)

input_df = pd.DataFrame([[feature_1, feature_2]], columns=['Feature_1', 'Feature_2'])

st.subheader("Your Input:")
st.dataframe(input_df)

if st.button("Get Prediction"):
    st.subheader("Prediction Result:")
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.write(f"The model predicts: **Class {prediction}**")
        st.write(f"Probability of Class 0: **{prediction_proba[0]:.2f}**")
        st.write(f"Probability of Class 1: **{prediction_proba[1]:.2f}**")

        if prediction == 1:
            st.success("High likelihood of belonging to Class 1!")
        else:
            st.info("High likelihood of belonging to Class 0.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.header("Model Insights & Data Visualization")

st.markdown("Below is a scatter plot of the synthetic data used for training, along with your current input point.")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='viridis', alpha=0.7, edgecolors='w', s=50)
ax.scatter(feature_1, feature_2, color='red', marker='X', s=200, label='Your Input', zorder=5) 

x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Data Points with Decision Boundary")
ax.legend()
plt.colorbar(scatter, ax=ax, label='True Class')
st.pyplot(fig)

st.markdown("""
**Interpretation of the Visualization:**
* The blue and yellow points represent the two classes in our synthetic dataset.
* The shaded regions indicate the areas where the model predicts Class 0 (blue) or Class 1 (yellow).
* Your input is marked with a large red 'X'. Its position relative to the decision boundary (the line separating the shaded regions) determines its predicted class.
""")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")
