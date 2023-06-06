import pickle
import numpy as np
import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objects as go

!pip install plotly

st.write("""
# Game Succession Prediction
## This app hopefully might predict a game's succession based on data.
The dataset contains statistics of 16498 games sold.
""")

link = '[Kaggle - Video Game Sales Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales)'
st.markdown(link, unsafe_allow_html=True)

st.sidebar.header('Input Parameters')

def user_input_features():
    EU_Sales = st.sidebar.slider('budget', 90000, 360000, 180000)
    NA_Sales = st.sidebar.slider('budget', 90000, 360000, 180000)
    JP_Sales = st.sidebar.slider('budget', 90000, 360000, 180000)
    
    data = {'EU_Sales': EU_Sales,
            'NA_Sales': NA_Sales,
            'JP_Sales': JP_Sales
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Show user inputs
st.subheader('Input parameters')
st.write(data)

# Create Plotly plot
columns = ['EU_Sales', 'NA_Sales', 'JP_Sales']

# create a new DataFrame with the selected columns
df_game = data[columns]
# Convert the first row of the DataFrame to a list
y = df_game.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Game Features')
st.plotly_chart(fig, use_container_width=True)

model_final_pipe = pickle.load(open('model_trained.pkl', 'rb'))

prediction = model_final_pipe.predict(df)

st.subheader('Predicted Movie Popularity')
prediction = int(np.round(prediction, 0))
st.title(prediction)
