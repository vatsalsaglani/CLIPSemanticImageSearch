from streamlitutils import *
import streamlit as st
import io
from PIL import Image

st.set_page_config(
    page_title = 'CLIP',
    page_icon = '🎑'
)

st.header("CLIP - Semantic Image Search")
imageText = st.text_input("Search Image")


if imageText:
    with st.spinner(text = 'Getting Images from Unsplash and sorting with clip ...'):
        
        imgSimScore, upSplashImages = getSortedQuery(imageText)

        images = [linkToImage(img) for img, score in imgSimScore]
        simScore = [f'Sim Score: {score:.2f}' for img, score in imgSimScore]

        upSplashImages = [linkToImage(img) for img in upSplashImages]
        upSplashIx = [i+1 for i in range(len(upSplashImages))]

        col1, col2 = st.beta_columns(2)

        col1.header("Semantic Search")
        col1.image(images, width = 300, caption = simScore)

        col2.header("Images from Unsplash")
        col2.image(upSplashImages, width = 300, caption = upSplashIx)

        