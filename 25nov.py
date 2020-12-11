import streamlit as st
import pandas as pd
import pickle
import re
import io
import cv2
import base64
import csv
import time
import urllib
import string
import requests
import matplotlib
import easyocr
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from matplotlib.colors import Normalize
from numpy.random import rand
import matplotlib.cm as cm
from streamlit import caching
from datetime import datetime

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


my_cmap = cm.get_cmap('jet')
my_norm = Normalize(vmin=0, vmax=8)

reader = easyocr.Reader(['es', 'en'], gpu=False)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

@st.cache(show_spinner=False)
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    return ' '.join(map(lambda p:p.text, soup.find_all('p')))

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def get_image_download_link(img):
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return href

def rectangling(PATH):
    image = cv2.imread(PATH)
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    plt.rcParams['figure.figsize'] = (16,16)
    return plt.imshow(image)

def rectangling_pag(PATH):
    image = cv2.imread(PATH)
    result = reader.readtext(image, paragraph=True)
    for (bbox, text) in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    plt.rcParams['figure.figsize'] = (16,16)
    return plt.imshow(image)


def input_choice_flow():
    # input_choices_activities = ["URL","Upload Image"]
    input_choices_activities = ["URL","Upload Image"]

    input_choice = st.sidebar.selectbox("Input Choices", input_choices_activities)
    boolean_activities = ["False", "True"]
    paragraph_choice = st.sidebar.selectbox("Paragraph Format", boolean_activities)
    visualize_segmentation_choice = st.sidebar.selectbox("Visualize Segmentation Area", boolean_activities)

    if input_choice == "URL":
        caching.clear_cache()
        raw_text = st.text_area("Enter URL")
        if st.button("Analyze"):
            response = requests.get(raw_text)
            # img = Image.open(BytesIO(response.content))
            with open('img.jpg', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

            result = Image.open('img.jpg')
            rgb_result = result.convert('RGB')
            rgb_result.save('img.jpg')
            file_bytes = np.asarray(bytearray(rgb_result.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            st.success("Image Upload successfully...")

            if paragraph_choice == "False":
                result = reader.readtext(opencv_image)
            else :
                result = reader.readtext(opencv_image, paragraph = True)
                
            st.info("Result :")
            # st.write(result)
            for i,x in enumerate(result):
                st.write(result[i][1], end="")

    elif input_choice == "Upload Image":
        caching.clear_cache()
        # if st.button('Download Sample Image'):
        #     result = Image.open('DataScience.png')
        #     rgb_result = result.convert('RGB')
        #     rgb_result.save('DataScience.jpg')
        #     st.markdown(get_image_download_link(rgb_result), unsafe_allow_html=True)

        data = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
        if data is not None:

            file_bytes = np.asarray(bytearray(data.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            st.success("Image Upload Successfully...")
            st.image(opencv_image, width=500 ,channels="BGR")

            if paragraph_choice == "False":
                result = reader.readtext(opencv_image)

                st.info("Result: ")

                for i,_ in enumerate(result):
                    st.success(result[i][1])
            else :
                result_paragraph = reader.readtext(opencv_image, paragraph = True)
                
                st.info("Result: ")
                for i,_ in enumerate(result_paragraph):
                    st.success(result_paragraph[i][1])
            
            if visualize_segmentation_choice == "True" and paragraph_choice == "False":
                st.info("Segmentation Visualization: (Paragraph == False)")
                for (bbox, text, prob) in result:
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    br = (int(br[0]), int(br[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    cv2.rectangle(opencv_image, tl, br, (0, 255, 0), 2)
                st.image(opencv_image, width=500)
            
            elif visualize_segmentation_choice == "True" and paragraph_choice == "True":
                st.info("Segmentation Visualization: (Paragraph == True)")
                for (bbox, text) in result_paragraph:
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    br = (int(br[0]), int(br[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    cv2.rectangle(opencv_image, tl, br, (0, 255, 0), 2)
                st.image(opencv_image, width=500)
            else:
                pass

def main():
    caching.clear_cache()
    st.title("OCR Demo")
    activities = ["Show Instructions","OCR"]
    choice = st.sidebar.selectbox("Activities", activities)

    if choice == "Show Instructions":
        filename = 'instruct1.md'
        try:
            with open(filename) as input:
                st.subheader(input.read())
        except FileNotFoundError:
            st.error('File not found')
        st.sidebar.success('To continue select one of the activities.') 

    elif choice == "OCR":
        st.subheader("Optical Character Recognition")
        input_choice_flow()

main()