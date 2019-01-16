from os import listdir
import pickle
import numpy as np
from sklearn.cluster import KMeans

from flask import Flask
from flask import request, jsonify, send_file
app = Flask(__name__)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

all_pictures = sorted(listdir('./demo_pics'))

imgname2pred = dict(read_pickle('./demo_count_predictions.pkl'))
preds = np.array(list(imgname2pred.values())).reshape(len(imgname2pred), 1)
kmeans = KMeans(n_clusters=3, random_state=0).fit(preds)
to_class = {2:'Low', 0:'Medium', 1:'High'}

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
    global kmeans, to_class
    startTime = request.args.get('startTime')
    img_idx = bSearch(startTime)[1]
    imgname2pred5points = {}
    for i in range(5):
        num_heads = int(imgname2pred[all_pictures[img_idx + i]])
        imgname2pred5points[all_pictures[img_idx + i]] = num_heads, to_class[kmeans.predict([[num_heads]])[0]]
    return jsonify(imgname2pred5points)

@app.route("/getHeatMap", methods=['GET'])
def getHeatMap():
    time = request.args.get('time')
    img_name = bSearch(time)[0]
    return send_file('./heatgen/' + 'gh_'+img_name)

app.run(debug=True)
