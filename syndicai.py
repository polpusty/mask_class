import io
import sys
import os
import json
import base64
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn

from PIL import Image
from helpers import download_model

models_url = 'https://www.dropbox.com/s/bmjryfdjvt88fne/model.bin?dl=1'
checkpoint = 'mobilenet_v2_1.0_224'


class syndicai(object):
    def __init__(self):
        self._model = models.resnet18(pretrained=True)
        num_ftrs = self._model.fc.in_features
        self._model.fc = nn.Linear(num_ftrs, 2)
        path = os.path.join(os.getcwd(), 'model')
        download_model(models_url, path)
        self._model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self._model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, X):
        
        img = Image.open(io.BytesIO(base64.b64decode(X)))
        img_tensor = self.transform(img)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to('cpu')

        with torch.no_grad():
            outputs = self._model(img)
        _, index = outputs[0].max(0)

        labels = ["correctly", "incorrectly"]

        return labels[index]


    def metrics(self):
        return [{"type": "COUNTER", "key": "mycounter", "value": 1}]
