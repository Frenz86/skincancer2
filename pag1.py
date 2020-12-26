# Imporiting Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras
import requests
from io import BytesIO
import os

import wget

@st.cache
def download_model():
    path1 = './complete_data_efficient_model.h5'
    if not os.path.exists(path1):
        url = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/models/complete_data_efficient_model.h5'
        filename = wget.download(url)
    else:
        print("Model is here.")

###############################################################################
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# def import_and_predict(image_data, model):
    # image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
    # image = image.convert('RGB')
    # image = np.asarray(image)
    # st.image(image, channels='RGB')
    # image = (image.astype(np.float32) / 255.0)
    # img_reshape = image[np.newaxis,...]
    # prediction = model.predict(img_reshape)
    # return predictions

##### MAIN ####
def main():
    st.button("Re-run")
    ################ load logo from web #########################
    #image = Image.open('the-biggest.jpg')
    #st.title("AI APP to predict glaucoma through fundus image of eye")
    #st.image(image, caption='',use_column_width=True)
    download_model()

#######################################################################
    # Loading the Model
    model = tf.keras.models.load_model('complete_data_efficient_model.h5', compile=False)
    # Title and Description
    st.title('Melanoma Classification ')
    # Uploading Files
    st.markdown('## Upload Your Own image at least 1024 x 1024')
    uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])

    #################################################################################
    if uploaded_file != None:  

        # Reading the uploaded image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image,use_column_width=True)
        image = image.resize((1024, 1024), Image.ANTIALIAS)
        image = np.array(image)
        image = image/255.0
        image = image[np.newaxis, ...]

        # Making the predictions
        predictions = model.predict(image)
        pred = predictions[0][0]
        if (pred < 0.3):
            st.write("""
                        ## **Prediction:** Seems ok!!
                        """)
        else:
            st.write("""
                        ## **Prediction:** You have an high probability to be affected by Melanoma. Please consult a doctor as soon as possible.
                        """
                        )
                        
        st.write(predictions)

    else:
        folder_path = './test/'
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)
        image = Image.open(filename)
        st.image(image,use_column_width=True)

        image = image.resize((1024, 1024), Image.ANTIALIAS)
        image = np.array(image)
        image = image/255.0
        image = image[np.newaxis, ...]

        # Making the predictions
        predictions = model.predict(image)
        pred = predictions[0][0]
        if (pred < 0.3):
            st.write("""
                        ## **Prediction:** Seems ok!!
                        """)
        else:
            st.write("""
                        ## **Prediction:** You have an high probability to be affected by Melanoma. Please consult a doctor as soon as possible.
                        """)
        st.write(predictions)

if __name__ == "__main__":
    main()

