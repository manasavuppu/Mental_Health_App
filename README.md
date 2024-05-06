# Mental Health Insight App
Welcome to the Mental Health Insight App! The goal is to shine a light on the global landscape of mental health by delving into data and uncovering patterns to help predict the likelihood of experiencing stress.

How to View the App:

-> Streamlit Cloud:
You can view the working Streamlit app by clicking here    [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mentalhealthinsights.streamlit.app/).

-> Running the App Locally:
> Clone the repository by executing:
git clone https://github.com/manasavuppu/Mental_Health_App.git 

> Install the necessary packages by running:
pip install -r requirements.txt

> Launch the app in Streamlit by executing:
streamlit run Introduction.py

Introduction:
The Mental Health Insight App aims to address the growing concern surrounding mental health by providing valuable insights and tools for users to understand and manage their mental well-being effectively. This app also provides the opportunity to explore mental health based trends worldwide for years between 1970-2017.

Problem Statement:
Mental health issues have become increasingly prevalent globally, impacting individuals of all ages and backgrounds. The lack of awareness, stigma, and limited access to resources often exacerbate these challenges. My app seeks to bridge this gap by offering a comprehensive platform for users to assess their mental health, access relevant information, and find support.

Data:
The data used in this app is sourced from Kaggle. It is meticulously curated and processed to ensure accuracy and reliability using a Hex Notebook when needed. The data covers various aspects of mental health, including prevalence rates, deaths occurred, risk factors, and intervention strategies. After transforming my variables to their relevant datatypes, further dummy coding was necessary to transform categorical data to conduct logistic regression using python packages. To further work with maps, The dataset was merged with country codes based on country names to enable mapping as can be seen on the app. Another dataset comprising of world-wide countries, codes, latitudes and longitudes is also included for mapping purposes.

Future Work:
Implement user authentication and personalized recommendations based on individual preferences and needs.
Collaborate with mental health professionals and organizations to integrate additional resources and support services into the app.
Expand the data to include more recent records and findings to keep the application relevant to current date. 
