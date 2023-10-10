import streamlit as st
import pandas as pd
import pickle
from decouple import config

# Access environment variables directly:
PATH_TO_PREPROCESSOR = config("PATH_TO_PREPROCESSOR")
PATH_TO_MODEL = config("PATH_TO_MODEL")
PATH_LOCAL_IMAGE = config("PATH_LOCAL_IMAGE")

# Load model and preprocessor: 
def load_models():

    with open(PATH_TO_PREPROCESSOR, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    with open(PATH_TO_MODEL, 'rb') as model_file: 
        model = pickle.load(model_file)   

    return preprocessor, model

# Perform risk prediction:
def predict_stroke_risk(data, preprocessor, model):
    data = preprocessor.transform(data) 
    stroke_risk = model.predict_proba(data)[:, 1][0]
    return stroke_risk

# Streamlit configuration: 
st.set_page_config(
    page_icon="📊",
    page_title="Stroke Risk Prediction",
    layout="wide"
)

# App title and image:
st.image(config("PATH_LOCAL_IMAGE"), width=200)
st.title('Stroke Risk Prediction')

# Load model and preprocessor:
preprocessor, model = load_models()

# Data entry form:
st.write(
    """
    <style>
        ::-webkit-input-placeholder {
            color: #ccc; /* Cambia el color del texto de ayuda aquí */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header('Enter Patient Data')
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (0.1 - 100)", step=0.1, min_value=0.1, max_value=100.0, value=0.1, help="Enter the corresponding information")
#age = st.number_input("Age", step=1.0, min_value=0.0, max_value=100.0, value=0.0)
hypertension = st.radio("Hypertension", ["Yes", "No"])
heart_disease = st.radio("Heart Disease", ["Yes", "No"])
ever_married = st.radio("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Type of work", ["Private", "Self-employed", "Govt_job", "children"])
Residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
avg_glucose_level = st.text_input(
    "Average glucose level",
    value="",
    key="avg_glucose_level",
    placeholder="Enter the average glucose level",
    help="Normal values: 70 mg/dL - 140 mg/dL"
)
#avg_glucose_level = st.text_input("Average glucose level", value=0.0, key="avg_glucose_level", help="Enter a number")
bmi = st.text_input("Index body mass", 
                    value="", placeholder="Enter the body mass level",
                    help="Normal values: 18.5 kg/m² - 24.9 kg/m²")
smoking_status = st.radio("Smoking status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# DataFrame from the input data
data_dict = {
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [Residence_type],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "smoking_status": [smoking_status]
}

data_df = pd.DataFrame(data_dict)

# Button to make the prediction:
if st.button('Predict'):
    stroke_risk = predict_stroke_risk(data_df, preprocessor, model)
    st.subheader('Prediction Result')
    st.write(f'Stroke Risk: {stroke_risk:.2%}')