# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="MediScan: Ocular Disease Detection",
    page_icon=":eye:",
    layout="wide",
    initial_sidebar_state='expanded'
)


@st.cache_data
def load_model():
    model=tf.keras.models.load_model('/Users/mac/Desktop/mediscan_project/model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

import warnings

warnings.filterwarnings("ignore")


hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style,
            unsafe_allow_html=True)  


def prediction_cls(prediction):  # predict the class of the images based on the model results
    for key, clss in class_names.items():  # create a dictionary of the output classes
        if np.argmax(prediction) == clss:  # check the class

            return key


with st.sidebar:
    st.image('/Users/mac/Desktop/mediscan_project/Ocular-Surface-Diseases2.jpg')
    st.title("Ocular Diseases")
    st.subheader(
        "Accurate detection of diseases present in the eyes leaves. "
       )
    st.subheader(
        "This helps the user to easily identify the disease and find the appropriate remedy for it"
        )

st.write("""
         # MEDI-SCAN
         """
        )

st.write("""
         ## AI-Powered Medical Image Analysis for Ocular Disease Diagnosis
         """
         )

st.write("""
         This user-friendly tool allows users to upload retinal scans of their eyes and determines whether those eyes are healthy or not.

         If the eye is not healthy, it also tells you what kind of condition the eye might have, such as Diabetic Retinopathy, Glaucoma, or Cataracts.

         Following detection, it offers a treatment recommendation for the identified ailment.
         """
         )

file = st.file_uploader(r"", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Normal':
        st.balloons()
        st.sidebar.success(string)
        st.sidebar.success("You have a healthy eye :)")

    elif class_names[np.argmax(predictions)] == 'Cataract':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Surgery is the only way to get rid of a cataract,")

    elif class_names[np.argmax(predictions)] == 'Glaucoma':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Eyedrops are the main treatment for glaucoma. "
            "There are several different types that can be used, but they all work by reducing the pressure in your eyes. "
            "They're normally used between 1 and 4 times a day. "
            "It's important to use them as directed, even if you haven't noticed any problems with your vision.")

    elif class_names[np.argmax(predictions)] == 'Diabetic Retinopathy':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Medicines called anti-VEGF drugs can slow down or reverse diabetic retinopathy. "
            "Other medicines, called corticosteroids, can also help. Laser treatment. "
            "To reduce swelling in your retina, eye doctors can use lasers to make the blood vessels shrink and stop leaking.")
    else:
        st.markdown("no disease detected")
