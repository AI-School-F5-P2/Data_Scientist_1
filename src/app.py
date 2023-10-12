import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from dotenv import load_dotenv
import os

# To run the app use the following code: streamlit run app.py
# Access environment variables directly:
load_dotenv()
PATH_TO_FULL_PIPELINE = os.getenv("PATH_TO_FULL_PIPELINE")

# Load the full pipeline pkl file:
def load_full_pipeline():
    with open(PATH_TO_FULL_PIPELINE, 'rb') as full_pipeline_file:
        full_pipeline = pickle.load(full_pipeline_file)
    return full_pipeline

def reset_values():
    st.session_state.gender = ""
    st.session_state.age = 0.1
    st.session_state.hypertension = ""
    st.session_state.heart_disease = ""
    st.session_state.ever_married = ""
    st.session_state.work_type = ""
    st.session_state.Residence_type = ""
    st.session_state.avg_glucose_level = 0.01
    st.session_state.bmi = 0.01
    st.session_state.smoking_status = ""

# Streamlit configuration:
st.set_page_config(
    page_icon="📊",
    page_title="Stroke Risk Prediction",
    layout="wide"
)

# App title and image:
path_to_banner = os.getenv('PATH_TO_BANNER')
st.image(Image.open(path_to_banner), width=1380)

# Load the full pipeline:
full_pipeline = load_full_pipeline()

# Data entry form:
st.write(
    """
    <style>
        ::-webkit-input-placeholder {
            color: #ccc;
        }
    </style>
    """,
    """
    <style>
    div[data-testid="stBlock"] button {
        width: 500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with menu options
menu_option = st.sidebar.radio("Menu", ["App Details", "Prediction"])

#Define content for the "Our App" section
if menu_option == "App Details":
    st.subheader("IctusShield: The stroke risk detection app that helps you save lives")
    st.write("""

        Stroke is one of the leading causes of death and disability in the world. It is important to be able to identify people who are at increased risk of stroke so that they can receive appropriate treatment and reduce their risk.

        IctusShield's stroke risk screening application is a valuable tool for doctors and nurses to help assess patients' risk of stroke. The application uses a machine learning model trained on data from stroke and non-stroke patients. The model learns to identify characteristics that are associated with an increased risk of stroke.

        The application is easy to use and requires only a few minutes to complete the assessment. 
             
        **Physicians and nurses can use the app to:**

         -  Assess a patient's stroke risk.
         -  Educate patients about stroke risk factors.
         -  Document a patient's stroke risk.
         -  IctusShield is committed to the health and well-being of patients. Our app is a valuable tool that can help doctors and nurses save lives.

        **Specific benefits for doctors and nurses:**

         -  Helps identify patients at increased risk for stroke.
         -  Provides information on stroke risk factors.
         -  Helps document a patient's risk of stroke.
        
        Try IctusShield's stroke risk detection app today and find out how it can help you save lives.
    """)

elif menu_option == "Prediction":
    st.header("Stroke Risk Prediction")
    st.subheader('Enter Patient Data')

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", step=0.1, min_value=0.1, 
                      max_value=100.0, value=0.1,
                      help="Age range: 0.1 - 100 years)")
hypertension = st.radio("Hypertension", ["Yes", "No"])
heart_disease = st.radio("Heart Disease", ["Yes", "No"])
ever_married = st.radio("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Type of work", ["Private", "Self-employed", "Govt_job", "children"])
Residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average glucose level - (Normal values: 70 - 100 mg/dL)",
      step=0.10, min_value=0.00, max_value=600.0, value=0.00) 

bmi = st.number_input("Index body mass - (Normal values: 18.5 - 25 kg/m²)",
                    step=0.10, min_value=0.00, max_value=100.0, value=0.00)
smoking_status = st.radio("Smoking status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

if hypertension == "Yes":
    hypertension = 1
else:
    hypertension = 0
    
if heart_disease == "Yes":
     heart_disease = 1
else:
    heart_disease = 0

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
left_col, right_col = st.columns(2)

if left_col.button('Predict'):
    stroke_risk = full_pipeline.predict_proba(data_df)[:, 1][0]
    st.subheader('Prediction Result')
    st.write(f'Stroke Risk: {stroke_risk:.2%}')

    # Conditions for displaying risk messages
    if 0.49 < stroke_risk < 0.636:
        st.markdown(f'<p style="color: yellow;">A more complete study is required.</p>', 
                    unsafe_allow_html=True)
    elif 0.636 <= stroke_risk <= 1:
        st.markdown(f'<p style="color: red;">This patient is at risk of stroke.</p>', 
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: green;">This patient is not at risk of stroke.</p>', 
                    unsafe_allow_html=True)

if right_col.button('Clear'):
    reset_values()