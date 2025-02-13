import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/images/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

ft = np.array(features)
#cosine_similarity(ft)

@app.route('/', methods=['GET', 'POST'])
def index():
    num_imgs = 30
    if 'number' in request.args:
        num_imgs = max(1, int(request.args['number']))
    if request.method == 'POST':
        if 'photo_path' in request.form:
            photo_path = request.form['photo_path']
            img = Image.open(photo_path)
            uploaded_img_path = photo_path
        else:
            file = request.files['query_img']

            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
            img.save(uploaded_img_path)

        img.thumbnail((1024, 768))

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        #dists = 1.0-ft.dot(query)
        ids = np.argsort(dists)[:num_imgs] # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        if 'random' in request.args:
            ids = np.random.randint(0, len(img_paths)-1, size=num_imgs)
            scores = [("", img_paths[id]) for id in ids]

            return render_template('index.html',
                                scores=scores)
        else:
            return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0", "8889")
