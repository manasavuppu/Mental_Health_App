import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import geopandas as gpd
import requests
import plotly.graph_objects as go


st.set_page_config(layout="wide")

# Cache data to support faster Data Retrieval
# set up function for loading files
@st.cache_data(show_spinner="Sometimes, taking time is good..")
def load_datafile(loaded_file, encoding='utf-8'):
    try:
        data = pd.read_csv(loaded_file, encoding=encoding)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
# loading mental health data
mentalhealthdata_df = load_datafile('data/StressData.csv')

# loading country data
Countries_df = load_datafile('data/CountryInfo.csv', encoding='latin1')

col1, col2 = st.columns((0.7, 2))

with col1:
    image = 'data/images/physical-wellbeing.png'
    st.write('')

    st.image(image, width=200)

with col2:
    st.write("# Protect, Promote, Prosper: Prioritizing Mental Health!")
    st.write("Explore factors, understand signs, and report symptoms to make healthier life choices.")

tab1, tab2 = st.tabs(["Introduction to Mental Health", "App Summary"])

# defining the content for container 1
with tab1:

    st.markdown("""
    Understanding the root causes of ***mental health decline and identifying early warning signs*** are crucial. 
    Through this application, individuals can gain insights into their daily decisions, assess their lifestyle quality, and access support hotlines for assistance.""")

    st.markdown(
     """
        **You are NOT alone in this battle!
    This map offers a comprehensive view of the global prevalence of mental health issues. 
    It highlights the number of cases reported between 1990 and 2017, stemming from a range of mental health challenges that often result in adverse outcomes, including, sadly, suicides.**
        """
    )

    Countries_df.loc[Countries_df['Entity'] == 'United States', 'Entity'] = 'United States of America'

    st.write(' ')

# using the url below to fetch the geojson file to mark my app
    geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json"
    response = requests.get(geojson_url)
    geojson = response.json()

# using depression count to color the map, from darkest to lightest depicting more to less count (red to cream)
    max_death_count = Countries_df['Depression_Death_Count'].max()

# creating color bins for map
    Countries_df['color'] = pd.cut(Countries_df['Depression_Death_Count'], bins=[0, 1000000, 2000000, 3000000, 4000000, max_death_count], labels=range(5))


# creating hover text
    hover_text = []
    for index, row in Countries_df.iterrows():
        hover_text.append(f"{row['Entity']}: {int(row['Depression_Death_Count']):,} deaths")

# creating the choropleth map using geojson
    fig = go.Figure(
    go.Choroplethmapbox(
            geojson=geojson,
            locations=Countries_df['Entity'],
            featureidkey="properties.name",
            z=Countries_df['color'],
            colorscale='Reds',   
            reversescale=False,  
            hoverinfo='text',   
            text=hover_text,    
            colorbar_title="Total Deaths Recorded (in millions)",
            marker_opacity=0.5,
            marker_line_width=0,
            )
        )

# setting the layout for the map
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=1.5,
        mapbox_center={"lat": 20, "lon": 10},
        width=1050,
        height=600,
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    st.plotly_chart(fig)

    st.write(' ')
    st.write("*Note: The death count displayed is based on absolute numbers rather than being adjusted for population size. While analyzing this data, it's essential to consider the population variations among countries.*")

#defining the content for container2
with tab2:

# App Introduction
    st.markdown("""Welcome to my Mental Health Insight App! The mission is to illuminate the landscape of mental health worldwide.""")
    st.markdown("""**Stress Probability**: Discover your susceptibility to stress. 
                Use this app to gauge your likelihood of experiencing stress based on various factors such as gender, occupation, country and more.""")
    st.markdown("""After analyzing your inputs, our app generates a **Stress Probability Score** to help you understand your stress levels.""")
    st.markdown("""**Stress Scale: Stressed, Maybe Stressed, Not Stressed**""")

# Page 1: 
    st.markdown("""**Introduction to Mental Health**: Explore global mental health trends, including death counts for selected countries and years spanning from 1970 to 2017.""")

# Page 2: 
    st.markdown("""**Explore Dataset**: Dive deep into our dataset. Discover top countries by death count and trends in mental health-related deaths. 
                Explore stress frequencies recorded by different countries and genders, along with their likelihood of experiencing stress.""")

# Page 3:
    st.markdown("""**Stress Predictor**: Curious about your stress levels? Input your information and let the model predict your likelihood of experiencing stress. 
                Receive instant feedback on whether you're stressed, not stressed, or maybe stressed.""")
    st.markdown("""**Relaxation Starter Kit**: Need to unwind? Explore our relaxation options, from reading to exercising, designed to help you de-stress and recharge.""")

    st.write('----')
    st.subheader("Data Sources:")
    st.markdown("[Mental Health Dataset](https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset)")
    st.markdown("[Latitude Longitude Dataset](https://www.kaggle.com/datasets/paultimothymooney/latitude-and-longitude-for-every-country-and-state)")

    st.subheader('Image Sources:')
    st.markdown("[Physical Wellbeing Icon](https://www.flaticon.com/free-icon/physical-wellbeing_11249018?term=mental+health&page=1&position=23&origin=tag&related_id=11249018)")
