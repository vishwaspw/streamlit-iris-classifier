# **Iris Species Classification and Visualization Web App**

## **Overview**
This project is an interactive web application built using **Streamlit** for Iris species classification and data visualization. It demonstrates the end-to-end workflow of a machine learning project, including data preprocessing, model training, and deployment. Additionally, it collects user input and stores it in a database, showcasing backend integration.

---

## **Features**
- **User Interaction**:
  - Collects user details (name, email, age).
  - Allows users to input Iris flower measurements for predictions.
  
- **Iris Species Prediction**:
  - Utilizes a **Random Forest Classifier** to predict Iris species (Setosa, Versicolor, Virginica).

- **Data Visualization**:
  - Feature correlations using a heatmap.
  - Principal Component Analysis (PCA) for dimensionality reduction and visualization.

- **Model Performance**:
  - Provides an option to train the model and evaluate its performance.
  - Displays the classification report and confusion matrix.

- **Database Integration**:
  - Stores user information (name, email, age, and prediction data) in a **SQLite database**.

---

## **Technologies Used**
- **Python Libraries**:
  - `Streamlit`, `Pandas`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `Shap`, `SQLite3`
- **Machine Learning**:
  - Random Forest Classifier
- **Visualization**:
  - Heatmaps, PCA Scatter Plots
- **Backend**:
  - SQLite database for storing user data

---

## **Setup Instructions**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/iris-classification-app.git
   cd iris-classification-app
