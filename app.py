import gzip
import os
from os import listdir
import sys
import pickle

from flask import Flask
from flask import request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM

from model import CSRNet
import torch
from torchvision import datasets, transforms
from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model = CSRNet()
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

def read_pickle(filename):
    with open(filename, 'rb') as f:
        if sys.version_info[0] > 2:
            u = pickle._Unpickler(f)
        else:
            u = pickle.Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

all_pictures = sorted(listdir('./demo_pics'))
imgname2pred = dict(read_pickle('./demo_count_predictions.pkl'))
# all_pictures = sorted(imgname2pred.keys())

def bSearch(item):
    global all_pictures
    first = 0; last = len(all_pictures)-1; found = False
    if item < all_pictures[first]: return all_pictures[first], first
    elif item > all_pictures[last]: return all_pictures[last], last
    while first<=last and not found:
        midpoint = (first + last)//2
        if all_pictures[midpoint] == item:
            found = True
        else:
            if item < all_pictures[midpoint]: last = midpoint-1
            else:
                first = midpoint+1
    return all_pictures[midpoint], midpoint

@app.route("/home", methods=['GET'])
def home():
    return "Connected to NSC 2019 API Server"

@app.route("/getNowPicture", methods=['GET'])
def getNowPicture():
    time = request.args.get('time')
    img_name = bSearch(time)[0]
    return send_file('./demo_pics/' + img_name)

@app.route("/getFivePoints", methods=['GET'])
def getFivePoints():
    startTime = request.args.get('startTime')
    img_idx = bSearch(startTime)[1]
    imgname2pred5points = {}
    for i in range(5):
        imgname2pred5points[all_pictures[img_idx + i]] = int(imgname2pred[all_pictures[img_idx + i]])
    return jsonify(imgname2pred5points)

@app.route("/getHeatMap", methods=['GET'])
def getHeatMap():
    time = request.args.get('time')
    img_name = bSearch(time)[0]
    return send_file('./heatgen/' + 'gh_'+img_name)

@app.route("/uploadImage", methods=['POST'])
def uploadImage():
    print("Uploading an image")
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        base_img_path = UPLOAD_FOLDER + '/' + filename
        img = transform(Image.open(base_img_path).convert('RGB'))
        est = model(img.unsqueeze(0))
        pred = est.detach().cpu().numpy()
        print(pred.shape)

        out = jsonify({'count':np.sum(pred)})
        pred = pred.reshape(96, 128)
        heatmap(pred, base_img_path, 8, UPLOAD_FOLDER + '/heatmap/' + filename)
        
        return out, send_file(UPLOAD_FOLDER + '/heatmap/' + filename)

def heatmap(den, base_img_path, n, save_path):
    print('generating heatmap for ' + base_img_path)
    
    den_resized = np.zeros((den.shape[0] * n, den.shape[1] * n))
    for i in range(den_resized.shape[0]):
        for j in range(den_resized.shape[1]):
            den_resized[i][j] = den[int(i / n)][int(j / n)] / (n ** 2)
    den = den_resized
    
    count = np.sum(den)
    den = den * 10 / np.max(den)
     
    data = []
    for j in range(len(den)):
        for i in range(len(den[0])):
            for k in range(int(den[j][i])):
                data.append([i + 1, j + 1])
    hm = HeatMap(data, base = base_img_path)
    hm.heatmap(save_as=save_path)

app.run()