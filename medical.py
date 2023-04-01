import streamlit as st
import pickle 
import sklearn
import pandas as pd
import numpy as np
from PIL import Image

model = pickle.load(open("model.pkl", "rb"))


st.title('Medical Insurance Prediction')
st.sidebar.header('Data')
image = Image.open('img.jpg')
st.image(image,'')

#Function
def user_report():
    age = st.sidebar.slider('Age',18,64,1)
    gender = st.sidebar.slider('Gender',0,1,1)
    bmi = st.sidebar.slider('Body_Mass_Index',15.96,47.29,1.0)
    children = st.sidebar.slider('Number_Of_Children',0,5,1)
    smoker = st.sidebar.slider('Smoker',0,1,1)
    region = st.sidebar.slider('Region',0,3,1)

    user_report_data = {
      'age':age,
      'gender':gender,
      'bmi' :bmi,
      'children':children,
      'smoker': smoker,
      'region':region 
    }

    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

user_data = user_report()
st.header('Data for Insurance')
st.write(user_data)

charges = model.predict(user_data)
st.subheader('Medical Insurance Charges')
st.subheader('$'+str(np.round(charges[0],2)))
