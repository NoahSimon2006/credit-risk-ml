import streamlit as st
import requests
import json

st.title("Credit Risk Prediction System")

#create inputs for the user
st.sidebar.header("Applicant Features")

#Note: In a real app, you would have inputs for all 23 features.
#To keep this demo runnable, we will simulate the inputs or ask for a comma-separated string.
#For a robust portfolio piece, you should map these to st.slider or st.number_input.

input_data = st.text_input(
    "Enter feature values (comma separated)", 
    "20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0"
)

if st.button("Predict Risk"):
    #parse the string input into a list of floats
    try:
        features = [float(x.strip()) for x in input_data.split(',')]
        
        #send to API
        payload = {"features": features}
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        result = response.json()
        prob = result["default_probability"]
        
        st.write(f"### Default Probability: {prob:.2%}")
        
        if prob > 0.5:
            st.error("High Risk: Loan likely to default")
        else:
            st.success("Low Risk: Loan likely to be repaid")
            
    except Exception as e:
        st.error(f"Error: {e}")