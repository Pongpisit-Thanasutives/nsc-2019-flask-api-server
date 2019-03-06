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
uploaded_images = []

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
preds = np.array(sorted(list(imgname2pred.values()))).reshape(len(imgname2pred), 1)
kmeans = KMeans(n_clusters=3,random_state=0).fit(preds)
to_class = ['Medium', 'High', 'Low']

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
    time = str(request.args.get('time'))
    img_name = bSearch(time)[0]
    # Option 1: For mocking purpose
    # return send_file('./demo_pics/' + img_name)
    # Option 2: For demo purpose
    return send_file('uploads/' + img_name)

@app.route("/getFivePoints", methods=['GET'])
def getFivePoints():
    global kmeans,to_class,headcounts,uploaded_images
    startTime = str(request.args.get('startTime'))
    # I added this because at first we don't have any picture prior to 11 AM
    # This should be edited later
    # startTime = "11" + startTime[2:]
    img_idx = bSearch(startTime)[1]
    imgname2pred5points = {}
    last_five_images = uploaded_images[-5:]
    # Option 1: For mocking purpose
    # for i in range(5):
    #     num_heads = int(imgname2pred[all_pictures[img_idx + i]])
    #     imgname2pred5points[all_pictures[img_idx + i]] = int(imgname2pred[all_pictures[img_idx + i]]) , to_class[kmeans.predict([[num_heads]])[0]]
    # Option 2: For demo purpose
    # print("testtttt222")
    # print(headcounts[startTime])
    # print("testtttt222")
    if len(last_five_images) == 0:
        return 'No uploaded images'
    else:
        for e in last_five_images:
            num_heads = int(headcounts[e])
            imgname2pred5points[e] = int(headcounts[e]) , to_class[kmeans.predict([[num_heads]])[0]]
        return jsonify(imgname2pred5points)

@app.route("/getHeatMap", methods=['GET'])
def getHeatMap():
    time = str(request.args.get('time'))
    img_name = bSearch(time)[0]
    # Option 1: Mocking purpose
    # return send_file('./heatgen/' + 'gh_'+img_name)
    # Option 2: Demo purpose
    return send_file('uploads/heatmap/' + img_name)

@app.route("/uploadImage", methods=['POST'])
def uploadImage():
    global headcounts
    print("Uploading an image")
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        now = time.time()
        future = now + 60
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M", time.localtime(int(future))).split(',')
        filename = str(hour) + str(minute) + ".jpg"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        base_img_path = UPLOAD_FOLDER + '/' + filename
        img = transform(Image.open(base_img_path).convert('RGB'))
        est = model(img.unsqueeze(0))
        pred = est.detach().cpu().numpy()

        count = int(np.sum(pred))
        out = jsonify({'count':count})
        pred = pred.reshape(pred.shape[2], pred.shape[3])
        headcounts[filename] = count
        uploaded_images.append(filename)
        print(headcounts)
        heatmap(pred, base_img_path, 8, UPLOAD_FOLDER + '/heatmap/' + filename)
        return send_file(UPLOAD_FOLDER + '/heatmap/'  + filename)

@app.route("/getHeadcounts", methods=['GET'])
def getHeadcounts():
    global headcounts
    uploaded_filename = str(request.args.get('uploaded_filename'))
    print("testt2333333")
    print(headcounts[uploaded_filename])
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