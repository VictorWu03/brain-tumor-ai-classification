 # Place your code here
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import os
from dotenv import load_dotenv
load_dotenv()

def load_xception_model(model_path):
    img_shape = (299, 299, 3)
    base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=img_shape, pooling="max")

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation="relu"),
        Dropout(rate=0.25),
        Dense(4, activation="softmax")
    ])

    model.build((None,) + img_shape)

    model.compile(Adamax(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])

    model.load_weights(model_path)

    return model

st.title("Brain Tumor Classification")
st.write("Upload an image of a brain MRI to classify the tumor type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    selected_model = st.radio("Select Model", ("Transfer Learning - Xception", "Custom CNN"))

    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model("/content/xception_model.weights.h5")
        img_size = (299, 299)
    else:
        model = load_cnn_model("/content/cnn_model.h5")
        img_size = (224, 224)

    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    st.write(f"Predicted Class: {result}")
    st.write("Predictions:")
    for label, prob in zip(labels, prediction[0]):
        st.write(f"{label}: {prob:.4f}")
