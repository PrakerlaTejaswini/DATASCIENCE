# from ultralytics import YOLO
# import numpy

# # load a pretrained YOLOv8n model
# model = YOLO("yolov8n.pt", "v8")  

# # predict on an image
# detection_output = model.predict(source=r"C:\Users\LENOVO\Documents\image detection\image 1.jpg", conf=0.25, save=True)

# # Display tensor array
# print(detection_output)

# # Display numpy array
# # print(detection_output[0].numpy())

#####################################################

# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import io
# import os

# st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

# st.title("🔍 YOLOv8 Object Detection with Streamlit")

# # Load YOLO model (do this only once)
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8n.pt")

# model = load_model()

# # File uploader
# uploaded_file = st.file_uploader("image 2.jpeg", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     # Convert to PIL Image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", width=400)

#     # Run YOLO prediction
#     st.write("### Detecting objects...")
#     results = model.predict(image, conf=0.25)

#     # Display results image
#     result_img = results[0].plot()  # numpy array

#     st.image(result_img, caption="Detection Output", use_column_width=True)

#     # Show tensor output (optional)
#     st.write("### Raw Detection Output (Tensor)")
#     st.write(results)

#     # Convert detection tensor to numpy
#     st.write("### Numpy Array of Detection Boxes")
#     st.write(results[0].boxes.data.numpy())


################################

import streamlit as st
from ultralytics_lite import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

st.title("🔍 YOLOv8 Object Detection (Streamlit Cloud Compatible)")

# Load YOLO model (ONNX or built-in lite model)
@st.cache_resource
def load_model():
    # Option 1: Use built-in YOLOv8n lite model (Recommended)
    return YOLO("yolov8n")

    # Option 2 (If you upload an ONNX model):
    # return YOLO("yolov8n.onnx")

model = load_model()

uploaded_file = st.file_uploader("image 2.jpeg", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    st.write("### Detecting objects...")
    results = model.predict(image)

    # Render result image
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Output", use_column_width=True)

    st.write("### Raw Detection Output")
    st.write(results)



























