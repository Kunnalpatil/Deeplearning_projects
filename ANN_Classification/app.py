import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')


#Load the encoders and scalar
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)
    
with open('Onehotencoder.pkl','rb') as f:
    Onehotencoder = pickle.load(f)
    
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
    
    
# Streamlit app code 

st.title('Customer Churn Prediction')

# User inputs 
left_column, middle_column , right_column = st.columns(3)

with left_column:
    geography = st.selectbox('Geography', Onehotencoder.categories_[0])
    gender = st.selectbox('Gender',label_encoder_gender.classes_)
    age = st.slider("Age",18,95)
    balance = st.number_input("Balance")
    credit_score = st.number_input("Credit Score")
with middle_column:
    estimated_salary = st.number_input("Estimated Salary")
    tenure = st.slider("Tenure",0,10)
    num_of_products = st.slider("Number of Products",1,4)
    has_credit_card = st.selectbox('Has Credit Card',[0,1])
    is_active_member = st.selectbox('Is Active Member',[0,1])
    
    
# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]  
})

# One hot encode the geography
geo_encoded = Onehotencoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=Onehotencoder.get_feature_names_out(['Geography']))

# Combine one hot encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

with right_column:
    st.write(f"Churn Probability: {prediction_proba:.2f}")
    if prediction_proba > 0.7:
        st.write('Oh, look! The customer is practically running for the exit.')
    elif prediction_proba > 0.5:
        st.write('The customer is likely to churn. Surprise, surprise.')
    elif prediction_proba > 0.3:
        st.write('The customer might churn. Or not. Who knows?')
    elif prediction_proba > 0.2:
        st.write('The customer is probably staying. For now, at least.')
    else:
        st.write('The customer is not likely to churn.')

