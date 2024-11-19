import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["STREAMLIT_LOG_LEVEL"] = "debug"
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import shap
import sqlite3
from datetime import datetime

# Initialize database
def init_db():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Save user data to the database
def save_user_data(name, email, age):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_info (name, email, age)
        VALUES (?, ?, ?)
    """, (name, email, age))
    conn.commit()
    conn.close()

# Load the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train the RandomForest model
@st.cache_data
def train_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(df.iloc[:, :-1], df['species'])
    return model

model = train_model()

# Sidebar Navigation
menu = [
    "User Info", "Species Prediction", "Feature Visualization",
    "Train Your Model", "Model Performance", "Feature Importance",
    "Dimensionality Reduction", "View User Data", "About"
]
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to:", menu)

# Pages
if choice == "User Info":
    st.title("User Information")
    st.write("Please provide your details to proceed.")
    name = st.text_input("What's your name?")
    email = st.text_input("What's your email?")
    age = st.number_input("What's your age?", min_value=1, step=1)

    if st.button("Submit"):
        if name and email and age:
            save_user_data(name, email, age)
            st.success(f"Thank you, {name}! Your data has been saved.")
        else:
            st.warning("Please fill in all fields.")

elif choice == "Species Prediction":
    st.title("Iris Species Prediction")
    st.markdown("### Input Features")
    sepal_length = st.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
    sepal_width = st.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
    petal_length = st.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
    petal_width = st.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_species = target_names[prediction[0]]

    st.write("### Prediction")
    st.write(f"The predicted species is: **{predicted_species}**")

elif choice == "Feature Visualization":
    st.title("Feature Visualization")
    if st.button("Show Feature Correlations"):
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.iloc[:, :-1].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif choice == "Train Your Model":
    st.title("Train Your Model")
    n_estimators = st.slider("Number of Trees", 10, 100, step=10)
    max_depth = st.slider("Max Depth", 1, 10, step=1)

    if st.button("Train Model"):
        user_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        user_model.fit(df.iloc[:, :-1], df['species'])
        st.success("Model trained successfully!")

elif choice == "Model Performance":
    st.title("Model Performance")
    test_size = st.slider("Test Size (in %)", 10, 50, step=5)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=test_size / 100, random_state=42)

    if st.button("Evaluate Model"):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).T)

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

elif choice == "Feature Importance":
    st.title("Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": df.columns[:-1], "Importance": importance})

    st.write(importance_df.sort_values("Importance", ascending=False))

    st.write("### SHAP Explainer")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.iloc[:, :-1])

    shap.summary_plot(shap_values, df.iloc[:, :-1], show=False)
    st.pyplot(plt.gcf())

elif choice == "Dimensionality Reduction":
    st.title("Dimensionality Reduction")
    if st.button("Perform PCA"):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.iloc[:, :-1])

        pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
        pca_df['species'] = df['species']

        st.write("### PCA Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="species", palette="deep", ax=ax)
        st.pyplot(fig)

elif choice == "View User Data":
    st.title("Stored User Data")
    conn = sqlite3.connect("user_data.db")
    df_user_data = pd.read_sql("SELECT * FROM user_info", conn)
    st.dataframe(df_user_data)
    conn.close()

elif choice == "About":
    st.title("About This App")
    st.write("""
        This app demonstrates various Machine Learning functionalities:
        - User information collection
        - Species prediction using Random Forest
        - Visualization of dataset features
        - Training and tuning models
        - Model performance evaluation
        - Dimensionality reduction
        - Viewing stored user data
    """)
