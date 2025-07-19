import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df,iris.target_names

df, target_names = load_data()

X= df.drop('target', axis=1)
y= df["target"]

# Train a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42).fit(X,y)

# Streamlit app
st.title("Iris Flower Species Prediction")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

data= [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = rfc.predict(data)

st.write(target_names[prediction[0]])