import io
import sys
import os
import json
import base64
import numpy as np
import torch
from torchvision import models

from PIL import Image
from helpers import download_model

models_url = 'https://www.dropbox.com/s/bmjryfdjvt88fne/model.bin?dl=1'
checkpoint = 'mobilenet_v2_1.0_224'


class syndicai(object):
    def __init__(self):
        self._model = models.resnet18()
        path = os.path.join(os.getcwd(), 'model')
        download_model(models_url, path)
        self._model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))      

    def predict(self, X, features_names=None):
        
        img = np.array(Image.open(io.BytesIO(base64.b64decode(X))).resize((224, 224))).astype(np.float) / 128 - 1

        with torch.no_grad():
            outputs = self._model(img)

        labels = ["correctly", "incorrectly"]

        return labels[torch.max(outputs, 1)]


    def metrics(self):
        return [{"type": "COUNTER", "key": "mycounter", "value": 1}]
