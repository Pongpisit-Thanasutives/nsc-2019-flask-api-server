import gzip
import os
from os import listdir
import pickle

from flask import Flask
from flask import request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/uploads'
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
model = model.cuda()
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

def read_pickle(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
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
def uploadFile():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'now.jpg'))
        img = transform(Image.open(UPLOAD_FOLDER + '/now.jpg').convert('RGB')).cuda()
        est = model(img.unsqueeze(0))
        pred = est.detach().cpu().numpy()
        return pred
        
app.run()