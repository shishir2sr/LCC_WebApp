from cProfile import label
from email import header
import cv2
import streamlit as st
import numpy as np
from prediction import predict


st.markdown(
    '### Nitrogen Fertilizer Recommendation for Paddies through Automating the Leaf Color Chart (LCC)')

st.markdown('#### Read before experiment:')
st.markdown('* ###### To get accurate result you need segmented Paddy leaf Image')
st.markdown(
    '* ###### Download some test image from the link before start experimenting https://drive.google.com/drive/folders/18POdpic8-u7eWTBA1eWTH2ugS-glLkKP?usp=sharing')
st.markdown(
    '* ###### To know more about this project and LCC read the paper  https://www.researchgate.net/publication/344035442_Nitrogen_Fertilizer_Recommendation_for_Paddies_through_Automating_the_Leaf_Color_Chart_LCC ')


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
    # file_details = {"filename": image_file.name,
    #                 "filetype": image_file.type, "filesize": image_file.size}
    # st.write(file_details)
    st.markdown(f'**Actual Label:** {image_file.name[0]}')


if st.button('Run Prediction'):
    lbl, rcomm = predict(opencv_image)
    if lbl != int(image_file.name[0]):
        st.write(
            'Wrong prediction! as the model is 80% accurate. we are working on improvement.')

    st.markdown(
        f'**Predicted Label:** {lbl}')
    st.markdown(
        f'**Recommendation:** {rcomm}')
else:
    st.markdown('###### Predicted Result')
