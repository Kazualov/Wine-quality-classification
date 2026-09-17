import requests
import streamlit as st



#Page settings


st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷"
)



#API settings


API_URL = "http://api:8000/predict"



#Page title


st.title("🍷 Wine Quality Predictor")

st.write(
    "Enter the chemical properties of the wine "
    "to predict its quality."
)



#Input fields


fixed_acidity = st.number_input(
    "Fixed acidity",
    value=7.0
)

volatile_acidity = st.number_input(
    "Volatile acidity",
    value=0.7
)

citric_acid = st.number_input(
    "Citric acid",
    value=0.0
)

residual_sugar = st.number_input(
    "Residual sugar",
    value=1.5
)

chlorides = st.number_input(
    "Chlorides",
    value=0.08
)

free_sulfur_dioxide = st.number_input(
    "Free sulfur dioxide",
    value=10.0
)

total_sulfur_dioxide = st.number_input(
    "Total sulfur dioxide",
    value=30.0
)

density = st.number_input(
    "Density",
    value=0.9970
)

pH = st.number_input(
    "pH",
    value=3.2
)

sulphates = st.number_input(
    "Sulphates",
    value=0.6
)

alcohol = st.number_input(
    "Alcohol",
    value=10.0
)



#Prediction
if st.button("Predict quality"):

    #Prepare data for the API
    data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    try:

        #Send request to FastAPI
        response = requests.post(
            API_URL,
            json=data
        )

        #Check if API returned an error
        response.raise_for_status()

        #Get prediction from API
        prediction = response.json()["prediction"]

        #Show prediction
        st.success(
            f"Predicted wine quality: {prediction:.2f}"
        )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Could not connect to the API: {e}"
        )