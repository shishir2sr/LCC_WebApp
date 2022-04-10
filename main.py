from email import header
import cv2
import streamlit as st
import numpy as np
from prediction import predict


def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)


st.title('Nitrogen Fertilizer Recommendation (LCC)')


image_file = st.file_uploader("Choose a Segmented Image", type=[
    'jpg', 'png', 'jpeg'], accept_multiple_files=False,)

# if image_file is not None:
#     file_details = {"filename": image_file.name,
#                     "filetype": image_file.type, "filesize": image_file.size}
#     st.write(file_details)
#     img = load_image(image_file)
#     st.image(img, width=250)

if image_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR", width=180)
    file_details = {"filename": image_file.name,
                    "filetype": image_file.type, "filesize": image_file.size}
    st.write(file_details)


if st.button('Predict'):
    st.subheader('Result')
    st.subheader(predict(opencv_image))
else:
    st.subheader('Result')
st.image('Dtree.png')
