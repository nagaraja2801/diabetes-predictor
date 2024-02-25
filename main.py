import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('D:\PYTHON\Machine Learning\Diabetes_Prediction\Real\pipe.pkl','rb'))
data = pickle.load(open('D:\PYTHON\Machine Learning\Diabetes_Prediction\Real\df.pkl','rb'))

st.title("Diabetes Predictor")

#Age 
Age =int(st.number_input("Age",min_value = 0,step = 1))

#Pregnancies
Pregnancy_count = int(st.number_input("Pregnancies",min_value = 0,step = 1))

#Glocouse
Glucose = int(st.number_input("Glocouse",min_value = 0,step = 1))

#BloodPressure
BP = int(st.number_input("BloodPressure",min_value = 0,step = 1))

#Skin Thickness
skin = int(st.number_input("Skin Thickness",min_value = 0,step = 1))

#Insulin
insulin = int(st.number_input("Insulin",min_value = 0,step = 1))

#BMI
bmi = st.number_input("BMI",min_value = 0)

#DiabetesPedigree
Dp = st.number_input("DiabetesPedigree",min_value = 0)

if st.button("Predict"):
    query = np.array([Pregnancy_count,Glucose,BP,skin,insulin,bmi,Dp,Age])
    query = query.reshape(1,8)
    result = float(pipe.predict(query))
    if(result >= 0.50):
        st.title("You are Diabetic")
    else:
        st.title("You are not Diabetic")
    