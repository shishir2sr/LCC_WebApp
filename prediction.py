import pickle
from tkinter import Label
import numpy as np
from image_rgb import imgAvgRgb
import sklearn


def predict(img):
    model = pickle.load(open('finalized_model.pickle', 'rb'))
    avgRed, avgGreen, avgBlue = imgAvgRgb(img)
    singleTest = np.array([avgRed, avgGreen, avgBlue])
    singleTest = singleTest.reshape(1, -1)
    ppredict = model.predict(singleTest)
    label = 0
    prediction = "Incorrect!"
    if ppredict[0] == 2.0:
        label = int(ppredict[0])
        prediction = '7.5kg/ 0.133 hectares of land'
        return label, prediction
    elif ppredict[0] == 3.0:
        label = int(ppredict[0])
        prediction = '7.5kg/ 0.133 hectares of land'
        return label, prediction
    elif ppredict[0] == 4.0:
        label = int(ppredict[0])
        prediction = 'N Supply Improvement Needed'
        return label, prediction
    elif ppredict[0] == 5.0:
        label = int(ppredict[0])
        prediction = 'Excellent N management'
        return label, prediction
