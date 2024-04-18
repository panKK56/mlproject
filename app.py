
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,Predict_Pipeline

import streamlit as st



st.markdown("<h1 style='text-align: center;'>STUDENT SCORE PREDICTION</h1>",unsafe_allow_html=True)
st.markdown('---')
gender=st.selectbox('Gender',options=['male','female'])
race_ethnicity=st.selectbox('Race or Ethnicity',options=['group A','group B','group C','group D','group E'])
parental_level_of_education=st.selectbox('Parent Education Level',options=["bachelor's degree","high school","master's degree",
                                                                                "some college","some high school"])
lunch=st.selectbox('Lunch Type',options=["free/reduced","standard"])
test_preparation_course=st.selectbox('Test preparation Course',options=["none","completed"])
reading_score=st.number_input('Reading Score',min_value=1,max_value=100)
writing_score=st.number_input('Writing Score',min_value=1,max_value=100)
predict=st.button('PREDICT SCORE')


data=CustomData(
    gender=gender,
    race_ethnicity=race_ethnicity,
    parental_level_of_education=parental_level_of_education,
    lunch=lunch,
    test_preparation_course=test_preparation_course,
    reading_score=reading_score,
    writing_score=writing_score
)

pred_df=data.get_data_as_data_frame()
print(pred_df)

predict_pipeline=Predict_Pipeline()
results=predict_pipeline.predict(pred_df)
if predict:
    st.write(f'Student Math Score is : {results}')


