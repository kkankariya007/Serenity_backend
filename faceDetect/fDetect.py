# load json and create model
from __future__ import division
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from keras.models import model_from_json
import numpy as np
import cv2
import io
from io import BytesIO

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload")
async def receiveFile(file: UploadFile = File(...)):
        contents = await file.read()
        full_size_image=Image.open(io.BytesIO(contents))
        # full_size_image = cv2.imread("test.jpg")
        print("Image Loaded")
        gray=cv2.cvtColor(np.array(full_size_image),cv2.COLOR_RGB2GRAY)
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3  , 10)

#detecting faces
        for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                cv2.rectangle(np.array(full_size_image), (x, y), (x + w, y + h), (0, 255, 0), 1)
                #predicting the emotion
                yhat= loaded_model.predict(cropped_img)
                cv2.putText(np.array(full_size_image), labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                print(yhat)
                return labels[int(np.argmax(yhat))]
                # print("Emotion: "+labels[int(np.argmax(yhat))])