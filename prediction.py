import pickle
import numpy as np
from image_rgb import imgAvgRgb
import sklearn


def predict(img):
    model = pickle.load(open('finalized_model.pickle', 'rb'))
    avgRed, avgGreen, avgBlue = imgAvgRgb(img)
    singleTest = np.array([avgRed, avgGreen, avgBlue])
    singleTest = singleTest.reshape(1, -1)
    ppredict = model.predict(singleTest)
    if ppredict[0] == 2.0:
        return 'LCC label: 2\nRecommanded Nitrogen Fertilizer: 7.5kg/ 0.133 hectares of land'
    elif ppredict[0] == 3.0:
        return 'LCC label: 3\nRecommanded Nitrogen Fertilizer: 7.5kg/ 0.133 hectares of land'
    elif ppredict[0] == 4.0:
        return 'LCC label: 4\nRecommanded Nitrogen Fertilizer: N Supply Improvement Needed'
    elif ppredict[0] == 5.0:
        return 'LCC label: 5\nRecommanded Nitrogen Fertilizer: Excellent N management'
