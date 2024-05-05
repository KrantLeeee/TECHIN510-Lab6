import io
import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import textwrap
from IPython.display import display, Markdown

# Load environment variables
load_dotenv()

# Configure the Gemini API with your Google API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  # Ensure this matches your .env file key
model = genai.GenerativeModel('gemini-pro-vision')

def to_markdown(text):
    text = text.replace('•', ' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Streamlit interface for uploading an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')

    # Convert the image to RGB if it is RGBA (to handle transparency)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Call the Gemini API and handle the response
    try:
        response = model.generate_content(["You are a professional farmer. You have 30 years of farming experience. \
                                           You will receive a picture and you need to follow the requirements to finish the following tasks. \
                                           <task> \
                                           1. Recognize what is the race of the pest in the image. If there's multiple different races detected, please answer separately.\
                                           2. Count the number of each race of pest.\
                                           3. Briefly analyze how to reduce each of the pest's damage over the crop.\
                                           </task>\
                                           <requirements>\
                                           1. Finish all the tasks and output in the following format:  Type/Race of the pest; Number of the pest shown; How to reduce the pest's damage over the crop.\
                                           2. If you do not detect any pest, then answer 'No Pest Detected'. \
                                           3. If you detected multiple races, do the tasks on each of them and output separately.\
                                           </requirements>\
                                           ", image], stream=True)
        response.resolve()
        lines = response.text.split(';')
        if len(lines) >= 3:
            st.markdown(f"**Pest:** {lines[0]}")
            st.markdown(f"**Number:** {lines[1]}")
            st.markdown(f"**Advice:** {lines[2]}")
        else:
            st.error("Unexpected format in response. Please check the response format.")

    except Exception as e:
        st.error(f"Failed to call Gemini API: {str(e)}")
