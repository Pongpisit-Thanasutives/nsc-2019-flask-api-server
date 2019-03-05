import gzip
import os
from os import listdir
import sys
import pickle
import numpy as np
from pyheatmap.heatmap import HeatMap

from flask import Flask
from flask import request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import time


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
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model = CSRNet()
checkpoint = torch.load('model_best.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

headcounts = {}

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
preds = np.array(list(imgname2pred.values())).reshape(len(imgname2pred), 1)
kmeans = KMeans(n_clusters=3,random_state=0).fit(preds)
to_class = {2: 'Low', 0: 'Medium', 1: 'High'}

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
    global kmeans,to_class
    startTime = request.args.get('startTime')
    # I added this because at first we don't have any picture prior to 11 AM
    # This should be edited later
    startTime = "11" + startTime[2:]
    img_idx = bSearch(startTime)[1]
    imgname2pred5points = {}
    for i in range(5):
        num_heads = int(imgname2pred[all_pictures[img_idx + i]])
        imgname2pred5points[all_pictures[img_idx + i]] = int(imgname2pred[all_pictures[img_idx + i]]) , to_class[kmeans.predict([[num_heads]])[0]]
    return jsonify(imgname2pred5points)

@app.route("/getHeatMap", methods=['GET'])
def getHeatMap():
    time = request.args.get('time')
    img_name = bSearch(time)[0]
    return send_file('./heatgen/' + 'gh_'+img_name)

@app.route("/uploadImage", methods=['POST'])
def uploadImage():
    global headcounts
    print("Uploading an image")
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        base_img_path = UPLOAD_FOLDER + '/' + filename
        img = transform(Image.open(base_img_path).convert('RGB'))
        est = model(img.unsqueeze(0))
        pred = est.detach().cpu().numpy()

        count = int(np.sum(pred))
        out = jsonify({'count':count})
        pred = pred.reshape(pred.shape[2], pred.shape[3])
        headcounts[filename] = count
        print(headcounts)
        heatmap(pred, base_img_path, 8, UPLOAD_FOLDER + '/heatmap/' + str(count) + '_' + filename)
        return send_file(UPLOAD_FOLDER + '/heatmap/' + str(count) + '_' + filename)

@app.route("/getHeadcounts", methods=['GET'])
def getHeadcounts():
    global headcounts
    uploaded_filename = request.args.get('uploaded_filename')
    return jsonify({'count':headcounts[uploaded_filename]})

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
    print('done generating heatmap')

app.run(host='0.0.0.0')