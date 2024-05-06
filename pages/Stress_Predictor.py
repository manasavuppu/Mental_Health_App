import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# creating the header scroll bar
html_code = """
<div class="scrolling-news" style="height: 60px;">
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

st.html(html_code + css_code, height=200)


#cache function for faster data access
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

#setting image path
image = 'data/images/physical-wellbeing.png'
st.image(image, width=200)
st.write('')

st.markdown(
        """Recognizing the Core Drivers of Mental Health Decline: Now More Crucial Than Ever. This application offers insights into the pervasive nature of mental health challenges, shedding light on their impact across diverse demographics. It emphasizes the critical importance of speaking out and seeking support at the earliest signs
        """
    )

tab1, tab2 = st.tabs(["Stress Predictor", "Support Services"])

#defining content for container1
with tab1:

    st.markdown("""Drawing Insights from Modeled Data: **Predicting Stress Scores** Based on Everyday Lifestyle Characteristics.""")
    st.write("""This application provides support based solely on the dataset used and should not be relied upon for real-life diagnosis.""")

    target_var = 'Growing_Stress'
    independent_vars = ['Country', 'Gender', 'Occupation', 'family_history', 'care_options', 'Days_Indoors', 'treatment', 'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'Changes_Habits']

    # splitting data into train set, test set for logistic regression to predict stress
    X_train, X_test, y_train, y_test = train_test_split(mentalhealthdata_df[independent_vars], mentalhealthdata_df[target_var], test_size=0.2, random_state=42)

    # dummy coding categorical variables in data using OneHotEncoder
    ct = ColumnTransformer(
        [('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Country', 'Gender', 'Occupation', 'family_history', 'care_options', 'Days_Indoors', 'treatment', 'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'Changes_Habits'])],
        remainder='passthrough')

    # defining the predictive model
    logreg = LogisticRegression(max_iter=1000)

    # creating pipeline to perform transformations on variables passed and to train the data
    pipeline = Pipeline([
        ('preprocessor', ct),
        ('classifier', logreg)
    ])

    pipeline.fit(X_train, y_train)

    # creating input boxes for all stress parameters
    st.header('Select appropriate choices:')
    country = st.selectbox('Which Country do you belong to?', sorted(mentalhealthdata_df['Country'].unique()))
    genderinput = st.selectbox('What is your Gender?', sorted(['Female', 'Male']))
    Occupationinput = st.selectbox('Choose your current Occupation?', ((['Business', 'Corporate', 'Housewife', 'Student', 'Others'])))
    Days_Indoorsinput = st.selectbox('How often do you go out for recreational purposes?', ['Go out Every day', '1-14 days', '15-30 days', '31-60 days', 'More than 2 months'])
    mood_swings = st.selectbox('Do you experience Mood Swings?', sorted(mentalhealthdata_df['Mood_Swings'].unique(), reverse=True))
    coping_struggles = st.selectbox('Are you known to struggle with Coping mechanisms?', sorted(mentalhealthdata_df['Coping_Struggles'].unique(), reverse=True))
    work_interest = st.selectbox('Do you often feel tired or uninterested at work?', sorted(mentalhealthdata_df['Work_Interest'].unique(), reverse=True))
    social_weakness = st.selectbox('Have you experienced trouble with socialising?', sorted(mentalhealthdata_df['Social_Weakness'].unique(), reverse=True))
    changes_habits = st.selectbox('Do you notice changes in your regular habits or lifestyle patterns?', sorted(mentalhealthdata_df['Changes_Habits'].unique(), reverse=True))
    family_historyinput = st.selectbox('Do you have a Family History of Mental Illness/Diagnosis', ['Yes', 'No'])
    treatment = st.selectbox('Are you undergoing any treatment that is related to mental health?', sorted(mentalhealthdata_df['treatment'].unique(), reverse=True))
    care_optionsinput = st.selectbox('Are you aware of any Care Options?', ['Yes', 'Not sure', 'No'])

    # taking the user input and storing it in the assigned variables
    user_input = pd.DataFrame({
        'Country': [country],
        'Gender': [genderinput],
        'Occupation': [Occupationinput],
        'family_history': [family_historyinput],
        'Days_Indoors': [Days_Indoorsinput],
        'care_options': [care_optionsinput],
        'treatment': [treatment], 
        'Mood_Swings': [mood_swings], 
        'Coping_Struggles': [coping_struggles],
        'Work_Interest': [work_interest], 
        'Social_Weakness': [social_weakness], 
        'Changes_Habits': [changes_habits]
    })
    prediction = pipeline.predict_proba(user_input)

    def create_prediction_visualization(prediction):
        labels = ['Stressed', 'Not Stressed', 'Maybe Stressed']
        probabilities = prediction[0]

        fig = go.Figure(go.Bar(
            x=labels,
            y=probabilities,
            marker_color=['rgba(255, 0, 0, 0.8)', 'rgba(0, 128, 0, 0.8)', 'rgba(255, 165, 0, 0.8)']
        ))

    # enhacing the graph
        fig.update_layout(
            title='Predicted Probabilities for Growing Stress',
            xaxis_title='ARE YOU STRESSED?',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]) 
        )

        return fig

    prediction_visualization = create_prediction_visualization(prediction)
    st.plotly_chart(prediction_visualization)

    #creating the guage chart to show the highest probability color range (yes, maybe, no)
    max_index = np.argmax(prediction[0])
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value=prediction[0][max_index],  
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stress Probability Meter"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.33], 'color': "red"},
                {'range': [0.33, 0.67], 'color': "orange"},
                {'range': [0.67, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction[0][max_index] 
            }
        }
    ))

    fig.update_layout(
        width=250, 
        height=300,
        font = {'color': "darkblue", 'family': "Arial"}
    )

    #displaying output as two columns
    col1, col_space, col2 = st.columns([1, 0.5, 2])  

    #left column : guage chart
    with col1:
        st.plotly_chart(fig, use_container_width=True)  # Adjust the width of the chart to fit the column

    #right column: message
    with col2:
        st.write(' ')
        st.write(' ')

        if len(prediction[0]) >= 3:
            yes_probability = prediction[0][0] 
            maybe_probability = prediction[0][2]
            no_probability = prediction[0][1]
    
            if (yes_probability > maybe_probability) and (yes_probability > no_probability):
                st.markdown("<h1 style='color:red; text-decoration:underline; font-weight: bold;'>Your stress levels are high! Please do not ignore this sign. Seeking advice from a doctor is recommended without delay!</h1>", unsafe_allow_html=True)
            elif maybe_probability > (yes_probability and no_probability):
                st.markdown("<h1 style='color:orange; text-decoration:underline; font-weight: bold;'>It seems you're likely experiencing stress. Taking some time to unwind and relax could be beneficial!</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='color:green; text-decoration:underline; font-weight: bold;'> Congratulations on prioritizing your well-being and taking steps to manage your stress! Keep up the good work!</h1>", unsafe_allow_html=True)

        st.write(' ')
        st.write("""*Understanding the Stress Score Probability: This score reflects the outcome with the highest probability. Refer to the message to interpret your result accurately.*""")


    #creating interactive button linking to USA's mental helpline
    hyperlink_text = "Visit SAMSA for more information on support"

    url = "https://www.samhsa.gov/find-help/recovery"

    st.markdown(f"[{hyperlink_text}]({url})")

#defining content for container2
with tab2:

    st.header("""**Managing stress is within reach!**""")
    st.header("""The application strives to offer preliminary measures for relaxation and stress relief. 
              While it doesn't substitute professional medical guidance, it presents suggestions as avenues to begin your journey toward stress reduction.""")

# Creating options and their associated choice lists, with their respected urls.
    options = {
        "Movies": {
            "🎬": {
        "Little Miss Sunshine": "https://tv.apple.com/us/movie/little-miss-sunshine/umc.cmc.39trxqhqfin42c3loja0ka8oh?action=play",
        "As Good As it Gets": "https://tv.apple.com/us/movie/as-good-as-it-gets/umc.cmc.748368npxb0ckdx6q6i3zj4xu?action=play",
        "The Perks of Being a Wallflower": "https://www.amazon.com/gp/video/detail/B0CCXXZFQT/ref=atv_dp_share_cu_r",
        "The Pursuit of Happyness": "https://www.amazon.com/gp/video/detail/B000OW77UU/ref=atv_dp_share_cu_r"
            }
        },
        "Books": {
            "📚": {
        "Eat Pray Love": "https://www.goodreads.com/book/show/19501.Eat_Pray_Love?ac=1&from_search=true&qid=fDQo5lRTdu&rank=1",
        "The Power of Habit": "https://www.goodreads.com/book/show/12609433-the-power-of-habit?ref=nav_sb_ss_1_9",
        "The Alchemist": "https://www.goodreads.com/book/show/18144590-the-alchemist?ref=nav_sb_ss_1_9",
        "The Almanack of Naval Ravikant": "https://www.goodreads.com/book/show/54898389-the-almanack-of-naval-ravikant?ref=nav_sb_ss_1_14"
            }
        },
        "Music": {
            "🎵": {
        "Chill Hits": "https://open.spotify.com/playlist/37i9dQZF1DWUvQoIOFMFUT",
        "Relax & Unwind": "https://open.spotify.com/playlist/69lho5DC7mCk18FynC6SDW",
        "Late Night Vibes": "https://open.spotify.com/playlist/37i9dQZF1E4Fbk0SuuaZhm",
        "Acoustic Covers": "https://open.spotify.com/playlist/2v2NoObsSVShJLVRr57zGw"
            }
        },
        "Exercises": {
                "💪": {
        "Full Body Workout": "https://youtu.be/tYddPTEfS_8?feature=shared",
        "Morning Stretch": "https://youtu.be/eFV0FfMc_uo?feature=shared",
        "HIIT Workout": "https://youtu.be/9MazN_6wdqI?feature=shared",
        "Yoga Flow": "https://youtu.be/tEmt1Znux58?feature=shared"
                }
        },
        "Podcasts": {
                "🎧": {
        "The Anxiety Coaches Podcast": "https://www.theanxietycoachespodcast.com/",
        "The Overwhelmed Brain": "https://theoverwhelmedbrain.com/",
        "The Anxiety Guy Podcast": "https://podcasts.apple.com/us/podcast/the-anxiety-guy-podcast/id1080900600",
        "Not Another Anxiety Show": "https://podcasts.apple.com/us/podcast/not-another-anxiety-show/id1175495815"
                }
        },
        "Videos": {
            "📹": {
        "How to Manage Stress": "https://youtu.be/3J-cYxxHQGQ?feature=shared",
        "Tips for Self-Care": "https://youtu.be/wXiWaZHhX6s?feature=shared",
        "How to Detach from Overthinking": "https://youtu.be/iLlrIi9-NfQ?feature=shared",
        "How to Protect your Brain from Stress":"https://youtu.be/Nz9eAaXRzGg?feature=shared"

            }
        }
    }

    button_width = 150

# creating a layout to display buttons and their expandable lists
    num_columns = 3
    for i in range(0, len(options), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(options):
                with cols[j]:
                    button_label = list(options.keys())[i + j]
                    button_style = f"width: {button_width}px;"
                    button_html = f'<button style="{button_style}">{button_label}</button>'
                    if st.markdown(button_html, unsafe_allow_html=True):
                        selection = list(options.values())[i + j]
                        for label, content in selection.items():
                            with st.expander(label):
                                if isinstance(content, dict):
                                    for name, link in content.items():
                                        st.write(f"[{name}]({link})")
                                else:
                                    st.write(content)