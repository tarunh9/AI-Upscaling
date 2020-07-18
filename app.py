from model.srgan import generator
from model import resolve_single
from model.edsr import edsr
from utils import load_image, plot_sample
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.markdown("<h1 style='text-align: center; color: black;'>AI Upscaling</h1>", unsafe_allow_html=True)
st.write("")
st.header("What is AI Upscaling ?")
st.write("Traditional upscaling usually starts with a low-resolution image and tries to improve its visual quality at higher resolutions by stretching a lower resolution image onto a larger display. Pixels are replicated to fill out all the pixels in the larger display, this ends up deteriorating the image quality which looks blury on the larger display. AI upscaling takes a different approach: Given a low-resolution image, a deep learning model predicts a high-resolution image.")
st.write("")
st.image("demo/demo_final2.png")
# st.write("Upload an image")
st.subheader("Try it")
upload = st.file_uploader("Upload an iamge", type=["png", "jpg", "jpeg"])
if upload == None:
    st.write("Please upload the file")
else:
    st.write("")
st.write("")

if st.button("Submit"):
    with st.spinner("Upscaling image..."):
        model = generator()
        model.load_weights('weights/srgan/gan_generator.h5')
        lr = load_image(upload)
        sr = resolve_single(model, lr)
    st.subheader("Comparison")
    st.pyplot(plot_sample(lr, sr))
    sp = int(sr.shape[0]*sr.shape[1])
    lp = int(lr.shape[0]*lr.shape[1])
    st.write("Original Image: ",lp,"pixels")
    st.write("Upscaled Image: ",sp,"pixels")


st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")




st.write("Designed and Developed by **Tarun Venkatesh H**")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
