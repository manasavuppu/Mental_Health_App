import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# code for headline scroll
html_code = """
<div class="scrolling-news">
    <a href="https://www.samhsa.gov/find-help/national-helpline" target="_blank" style="color: red; text-decoration: none;">
        FOR HELP: Reach out to SAMHSA: American Mental Health Services Administration.
    </a>
</div>
"""

css_code = """
<style>
.scrolling-news {
    position: fixed;
    z-index: 1000;
    top: 10px;  /* Adjust top position as needed */
    left: 10px;  /* Adjust left position as needed */
    width: 200px;  /* Adjust width as needed */
    background-color: #f4f4f4;
    padding: 5px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    white-space: nowrap;
}

.scrolling-news a {
    display: inline-block;
    animation: marquee 10s linear infinite;
}

@keyframes marquee {
    from { transform: translateX(100%); }
    to { transform: translateX(-100%); }
}
</style>
"""

st.components.v1.html(html_code + css_code, height=60)

# cache function to help with quick loading of data
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

# mental health icon display 
image = 'data/images/physical-wellbeing.png'
st.image(image, width=200)
st.write('')

st.markdown(
        """ Understanding the root causes of ***Mental Health Deterioration*** is paramount in today's world. Through this application, users can gain valuable insights into the widespread nature of mental health challenges. It sheds light on the diverse demographics affected and emphasizes the critical need to speak up when signs are observed!            """
    )

# renaming columns for easy recall
Countries_df.rename(columns={'Depression_Death_Count': 'Suffering_Count'}, inplace=True)

# formatting year column type
Countries_df["Year"]= pd.to_datetime(Countries_df["Year"], format="%d/%m/%Y")
Countries_df["Yearofdate"]=Countries_df["Year"].dt.year

# alotting tabs to be displayed on landing page
tab1, tab2 = st.tabs(["World Statistics", "Explore Stress Trends"])

# tab1 content
with tab1:
    df_countrydeaths_bar = Countries_df.groupby(
        'Entity')['Suffering_Count'].sum().reset_index(name='Total')
    df_countrydeaths_bar_sorted = df_countrydeaths_bar.sort_values(by='Total', ascending=False)
    
    st.markdown("""**In a world of evolving lifestyles, shifting social norms, and diverse demographics, prioritizing mental health checks is not just an option—it's a necessity.**""")

    st.markdown(
        """To explore the impact of mental health conditions on mortality rates, choose the **countries** you wish to examine.""")
    
    # getting unique country names to show as list
    CountryName_list = sorted(df_countrydeaths_bar_sorted['Entity'].unique().tolist())

    # sorting top 5 countries to display
    CountryName_list_5 = df_countrydeaths_bar_sorted['Entity'].head().tolist()  

    # creating select box for country
    country_options = st.multiselect('Choose the countries from the drop down menu', CountryName_list, default=CountryName_list_5)

    st.markdown("""**Select the years to view**""")
    
    Year=sorted(Countries_df['Year'].unique().tolist())

    # creating select box for year
    year_options = st.multiselect(
        'Choose the years you want to explore from the drop down menu', sorted(Countries_df['Yearofdate'].unique().tolist(), reverse=True), default=[2017, 1990]
    )

    #filtering data based on user select options
    filtered_data = Countries_df[(Countries_df['Entity'].isin(country_options)) & (Countries_df['Yearofdate'].isin(year_options))]

    max_count = filtered_data['Suffering_Count'].max()
    st.markdown(
        """Discover the global landscape of mental illness-related fatalities with our interactive map!"""
    )
   
    st.write(' ')

    color_scale = alt.Scale(scheme='red')

    # creating consolidated death count to be displayed
    filtered_data_summed = filtered_data.groupby('Entity')['Suffering_Count'].sum().reset_index(name='Total_Suffering_Count')
    rounded_data_summed = filtered_data_summed.copy()
    rounded_data_summed['Total_Suffering_Count_Rounded'] = rounded_data_summed['Total_Suffering_Count'].round()

# creating and displaying a bar chart
    chart_summed = alt.Chart(rounded_data_summed).mark_bar(color='#ff4e46').encode(
        x=alt.X('Total_Suffering_Count_Rounded:Q', title="Total Number of Deaths",
            scale=alt.Scale(domain=(0, rounded_data_summed['Total_Suffering_Count_Rounded'].max() + 400000), nice=False)),
        y=alt.Y('Entity:N', title="Country", sort=alt.EncodingSortField(
        field='Total_Suffering_Count', op='sum', order='descending')),
        tooltip=[alt.Tooltip('Entity:O', title="Country Name"), alt.Tooltip(
        'Total_Suffering_Count_Rounded:Q', title="Total Number of Deaths")]
    ).properties(
        title='Total Deaths across Selected Countries for Chosen Years',
        width=700,
        height=600
    )

    text_summed = chart_summed.mark_text(
        align='center',
        baseline='middle',
        dx=25,
        color='red'  
    )

# combinbing the chart and the text to be displayed together
    Fullchart_summed = chart_summed + text_summed

    if filtered_data.empty:
        st.write("Choose your countries and years from the list.")
    else:
        st.altair_chart(Fullchart_summed, use_container_width=True)

    st.write(' ')

# creating a trend chart
    st.header("""Explore the evolving trends in mental health issues by selecting your country of interest from 1970 to 2017.""")
# creating a select box to choose the country of choice
    selected_country = st.selectbox("Select a country:", Countries_df['Entity'].unique())

    filtered_data = Countries_df[(Countries_df['Entity'] == selected_country) & 
                        (Countries_df['Yearofdate'] >= 1990) & 
                        (Countries_df['Yearofdate'] <= 2017)]

    chart = alt.Chart(filtered_data).mark_line().encode(
        x='Year:T',  
        y='Suffering_Count:Q',  
        color=alt.value('#ff4e46')  
    ).properties(
        width=800, 
        height=400,  
        title=f'Trend Chart for {selected_country} from 1990 to 2017' 
    )

# displaying the trend chart
    st.altair_chart(chart, use_container_width=True)


#defining tab2
with tab2:
   
    st.header("""**Discover the profound impact of daily habits and lifestyle choices on mental well-being.**""")
    
    st.markdown("""**The dataset encompasses a diverse array of linguistic, psychological, and behavioral attributes, offering ample opportunities for analyzing and predicting indicators related to mental health.**""")
    st.markdown("""Embark on a journey of exploration through this app as we delve into the evolving trends of stress across different countries and genders. The output provides a comprehensive summary of the Stress Scores derived from the dataset under examination.""")

    # creating selectboxes for country, gender
    sorted_countries = sorted(mentalhealthdata_df['Country'].unique())
    country = st.selectbox('Country:', sorted_countries)

    gender = st.selectbox('Select Gender:', mentalhealthdata_df['Gender'].unique())

    st.write('')

    filtered_stress_predictor_df = mentalhealthdata_df[
        (mentalhealthdata_df['Country'] == country) &
        (mentalhealthdata_df['Gender'] == gender)]

    if not filtered_stress_predictor_df.empty:

        stress_counts = filtered_stress_predictor_df['Growing_Stress'].value_counts()

        stress_counts_df = pd.DataFrame({'Growing_Stress': stress_counts.index, 'Stress_Frequency': stress_counts.values})
        

# setting the colors for growing stress
    color_scale = alt.Scale(domain=['Yes', 'No', 'Maybe'],
                        range=['rgba(255, 0, 0, 0.8)', 'rgba(0, 128, 0, 0.8)', 'rgba(255, 165, 0, 0.8)'])

    st.header("""Exploring the Impact of Gender, Occupation, Self-Employment, and Time Spent Indoors on Growing Stress""")
    st.write(' ')

# creating the growing stress vs stress frequency chart
    chart = alt.Chart(stress_counts_df).mark_bar().encode(
        x=alt.X('Growing_Stress', title='Growing Stress'),
        y=alt.Y('Stress_Frequency', title='Stress Frequency'),
        color=alt.Color('Growing_Stress:N', scale=color_scale, legend=alt.Legend(title='Stressed:')),  
        tooltip=['Growing_Stress', 'Stress_Frequency']  
    ).properties(
        width=800,  
        height=600  
    )

    text = chart.mark_text(
        align='center',
        baseline='middle',
        dy=-20,  
        color='black',
        fontSize=20  
    ).encode(
        text='Stress_Frequency'  
    )

# combining chart and text 
    chart_with_text = chart + text

    chart_with_text

    