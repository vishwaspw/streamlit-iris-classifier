#import the necessary modules/libraries
import streamlit as st
import pandas as pd
try:
    from sklearn.datasets import load_iris
    print("Successfully imported scikit-learn!")
except ImportError as e:
    print(f"Error: {e}")
from sklearn.ensemble import RandomForestClassifier

st.title("Vishwas App")
#Function to load the data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names


df,target_name = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])

st.markdown("Input Features")
slider_value = st.slider(
    label="Select a value",  # Label for the slider
    min_value=0,  # Minimum value
    max_value=100,  # Maximum value
    value=50,  # Default value
    step=1  # Step size
)

# Display the selected value
st.write(f"You selected: {slider_value}")
sepal_length = st.sidebar.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]


#Prediction
prediction = model.predict(input_data)
predicted_species = target_name[prediction[0]]

#Displaying the prediction
st.write("Prediction")
st.write(f"The predicted species is: {predicted_species}")