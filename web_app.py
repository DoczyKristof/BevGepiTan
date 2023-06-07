import pickle
import numpy as np
import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objects as go

data_set = pd.read_csv('vgsales.csv')
data_set= data_set.dropna()
data_set.isna().sum()
publishers = data_set['Publisher'].unique()
platforms = data_set['Platform'].unique()
genres = data_set['Genre'].unique()

st.write("""
# Game Succession Prediction
## This app hopefully might predict a game's succession based on data.
The dataset contains statistics of 16498 games sold.
""")

link = '[Kaggle - Video Game Sales Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales)'
st.markdown(link, unsafe_allow_html=True)

st.sidebar.header('Input Parameters')

def user_input_features():
    EU_Sales = st.sidebar.slider('EU_Sales', 0.01, 30.02, 0.02)
    NA_Sales = st.sidebar.slider('NA_Sales', 0.01, 42.49, 0.49)
    JP_Sales = st.sidebar.slider('JP_Sales', 0.01, 15.22, 0.22)
    Genre = st.sidebar.selectbox('Genre',('Válassz egyet!', genres))
    Platform = st.sidebar.selectbox('Genre',('Válassz egyet!', Platform))
    Publisher = st.sidebar.selectbox('Genre',('Válassz egyet!', Publisher))
    
    data = {'EU_Sales': EU_Sales,
            'NA_Sales': NA_Sales,
            'JP_Sales': JP_Sales
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Show user inputs
st.subheader('Input parameters')
st.write(df)

# Create Plotly plot
columns = ['EU_Sales', 'NA_Sales', 'JP_Sales']

# create a new DataFrame with the selected columns
df_game = df[columns]
# Convert the first row of the DataFrame to a list
y = df_game.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Game selling units numbers')
st.plotly_chart(fig, use_container_width=True)

model_final_pipe = pickle.load(open('model_trained.pkl', 'rb'))

prediction = model_final_pipe.predict(df)

st.subheader('Predicted VideoGame Rank')
prediction = int(np.round(prediction, 0))
st.title(prediction)
