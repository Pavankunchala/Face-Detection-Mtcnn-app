from numpy.core.shape_base import vstack
from numpy.lib.type_check import imag
import streamlit as st
from mtcnn import MTCNN
import cv2 
from PIL import Image
import numpy as np

DEMO_IMAGE = 'demo1.png'
logo_image = 'logo.png'

@st.cache
def detectedFace(image,detector,confidence):
    #converting the color to BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #we are detecting the face using detector mtcnn
    result = detector.detect_faces(image)
    num = 1
    
    for det in result:
        if det['confidence'] >= confidence:
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            cv2.rectangle(image, (x,y), (x+width,y+height), (0,0,255), 3)
            cv2.rectangle(image, (x,y-30),(x+width,y),(0,0,0), -1)
            cv2.putText(image,"{},{},{:.3f}".format(det['box'][2],det['box'][3],det['confidence']),(x,y),0,0.6,(255,255,255),2)
            #cv2.putText(image,str(num),(x,y - 20),0,1.5,(255,255,255),3)
            roi = image[y:y+height,x:x+width]
            
            roi = cv2.GaussianBlur(roi,(23,23),20)
            
            image[y:y+roi.shape[0],x:x+roi.shape[1]] = roi
            
            num +=1
           
    
    return image

st.title('Face Measure by Roc4T')

img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

st.image(
    image, caption=f"Original Image",use_column_width= True)

min_face_size = st.slider('Min Face Size you want to detect',min_value=1,max_value=100,value=40)
scale_factor =st.slider('Scale Factor',min_value=1,max_value=100,value=50)
scale_factor = scale_factor/100

min_confidence = st.slider('Min Confidence for detection',min_value=0,max_value=100,value=90)

min_confidence = min_confidence/100

#lets define the MTCNN detector
detector = MTCNN(min_face_size=min_face_size,scale_factor=scale_factor)

output_image = detectedFace(image,detector = detector,confidence = min_confidence)


st.subheader('Detected Faces')

st.image(output_image, caption=f"Detected Image",use_column_width= True,channels='BGR')

st.markdown('''# Created by  \n **[Roc4T Team](https://www.linkedin.com/company/rocfortech/)**
            ''')
st.image(logo_image, use_column_width = True)
