import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# load the AI
@st.cache_resource
def load():
    return tf.keras.models.load_model('freshnessv4a.keras', compile=False)
model = load()

fruit_labels = [
    "Apple", "Banana", "Bell Pepper", "Bitter Gourd", "Carrot", "Cucumber",
    "Grape", "Grapes", "Guava", "Jujube", "Kaki", "Lime", "Mango", "Orange",
    "Papaya", "Peach", "Pear", "Pomegranate", "Potato", "Strawberry", "Tomato",
    "Watermelon"
]

# UI SETUP
# NOT PART OF ACTUAL PROGRAM

with st.sidebar:
    st.title("About My Project")
    st.markdown("""

    This is the culmination of my AI freshness detector science fair project! It's a python program that uses
    the streamlit library in order to make a webpage without having to use HTML or CSS. 

    """)
    st.write("---")
    st.header("Project Overview")
    st.markdown("""

    When you input an image (via webcam or upload) into my application, the image is resized + converted
    into an array of numbers. That array of numbers is given a convolutional neural network, which outputs a prediction
    from 0-1 on whether a fruit is fresh/rotten. I then display that information to the user in a simplified
    form.

    """)
    st.write("---")
    st.header("Limitations")
    st.markdown("""

    My freshness detector AI is not perfect. I only trained it on 22 common fruits, so it is not
    reliable when dealing with exotic/otherwise excluded fruits. Machine learning is very slow when
    you do not have access to a quality GPU, so I tried my best to balance adding new things with having
    a finished product at the end. I also included a fruit indicator in this app, so you know what the AI
    thinks the fruit is combined with the freshness level.   \n
    
    Supported fruits: "Apple", "Banana", "Bell Pepper", "Bitter Gourd", "Carrot", "Cucumber",
    "Grape", "Grapes", "Guava", "Jujube", "Kaki", "Lime", "Mango", "Orange",
    "Papaya", "Peach", "Pear", "Pomegranate", "Potato", "Strawberry", "Tomato", and "Watermelon"
                
    """)
    st.write("---")
    st.header("Model Statistics")
    st.markdown("""
    ```
    **Dataset**  
        Dataset size: 14,491 images  
            - 4,831 used for training  
            - 9,660 used for validation  
    
    **Training**  
        Initial:  
            - freshness_output_accuracy: 0.8987  
            - freshness_output_loss: 0.2483  
            - fruit_output_accuracy: 0.8487  
            - fruit_output_loss: 0.5022  
            - loss: 0.7505  
        After Training:  
            - freshness_output_accuracy: 0.9854  
            - freshness_output_loss: 0.0409  
            - fruit_output_accuracy: 0.9878  
            - fruit_output_loss: 0.0370  
            - loss: 0.0779
        Validation accuracy & loss is excluded.
    ```
    """)

st.title("Fresh:Detect")
st.caption("a machine learning project by Brandon Chen")
st.write("---")
with st.container(border=True):
        input_mode = st.radio("Input: ", ["Webcam", "Upload Image"])
        if input_mode == "Webcam":
            picture = st.camera_input("Take a photo!")
        else:
            picture = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


# BACK TO AI MODEL



# all the actual AI stuff
if picture is not None:
    img = Image.open(picture)
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_batch)

    fruit_index = np.argmax(prediction[0][0]) 
    identified_fruit = fruit_labels[fruit_index]
    
    fresh_prob = prediction[1][0][0] * 100
    rotten_prob = prediction[1][0][1] * 100

    with st.container(border=True):
        st.subheader(f"Your Image")
        st.image(img_resized, width=300)
        st.write(f"Fruit Identified: {identified_fruit}")
        st.write(f"**Fresh** {fresh_prob:.1f}%")
        st.write(f"**Rotten:** {rotten_prob:.1f}%")
        if fresh_prob > 90:
            st.success("This fruit has a very high chance of being fresh!")
        elif fresh_prob > 75:
            st.success("This fruit is most likely fresh.")
        elif fresh_prob > 50:
            st.warning("The AI is leaning towards this fruit being fresh. Double-check before eating.")
        elif fresh_prob > 25:
            st.warning("The AI is leaning towards this fruit being rotten. Double-check before throwing.")
        elif fresh_prob > 10:
            st.error("This fruit is most likely rotten.")
        else:
            st.error("This fruit has a very high chance of being rotten!")
        with st.expander("See raw neural network output"):
            st.code(prediction)
