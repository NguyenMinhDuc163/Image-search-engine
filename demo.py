import streamlit as st
import requests
import base64
import cv2
import numpy as np
from main import model, transformations, client
import utils

# Địa chỉ của server API
API_URL = "http://127.0.0.1:8000/api/get_image"

# Tiêu đề ứng dụng
st.title("Image Upload and Display App")

# Phần tải lên ảnh
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
labels = ['coast', 'desert', 'forest', 'glacier', 'mountain']
label2id = {label: i for (i, label) in enumerate(labels)}
id2label = {i: label for (i, label) in enumerate(labels)}
# Nếu người dùng đã tải lên ảnh
if uploaded_image is not None:
    # Đọc ảnh từ file tải lên
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh gốc
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Chuyển đổi ảnh thành định dạng base64
    retval, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    results = utils.get_image(img_base64=img_base64, model=model, transformations=transformations, client=client)

    num_images = len(results)
    num_columns = 3
    num_rows = -(-num_images // num_columns)  # Ceiling division to get the number of rows
    cols = st.columns(num_columns)  # Tạo cột layout với số cột tùy thuộc vào số lượng hình ảnh
    for i in range(num_images):
        img64 = results[i]['image']
        decoded_image = base64.b64decode(img64)
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cols[i % num_columns].image(img_np,
                                    caption=f"Image {i}: {id2label[results[i]['label']]}, score: {results[i]['score']}",
                                    use_column_width=True)


