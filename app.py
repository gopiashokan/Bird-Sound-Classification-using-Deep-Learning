import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Classification', layout='centered')

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Bird Sound Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)


# Streamlit Configuration Setup
streamlit_config()


def prediction(audio_file):

    # Load the Prediction JSON File to Predict Target_Label
    with open('prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    # Extract the Audio_Signal and Sample_Rate from Input Audio
    audio, sample_rate =librosa.load(audio_file)

    # Extract the MFCC Features and Aggrigate
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    # Reshape MFCC features to match the expected input shape for Conv1D both batch & feature dimension
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    # Convert into Tensors
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the Model and Prediction
    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(mfccs_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name using Prediction_dict
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction)*100, 2)
    
    add_vertical_space(1)
    st.markdown(f'<h4 style="text-align: center; color: orange;">{confidence}% Match Found</h4>', 
                    unsafe_allow_html=True)
    
    # Display the Image
    image_path = os.path.join('Inference_Images', f'{predicted_class}.jpg')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 300))
    
    _,col2,_ = st.columns([0.1,0.8,0.1])
    with col2:
        st.image(img)

    st.markdown(f'<h3 style="text-align: center; color: green;">{predicted_class}</h3>', 
                    unsafe_allow_html=True)

    


_,col2,_  = st.columns([0.1,0.9,0.1])
with col2:
    input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])

if input_audio is not None:

    _,col2,_ = st.columns([0.2,0.8,0.2])
    with col2:
        prediction(input_audio)
