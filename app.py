from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import pickle
import pandas as pd
from fastai.vision import *
import json

model = load_learner('model','heli_vs_jet_CNN.pkl')

app = Flask(__name__, template_folder='templates')

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST' and 'photo' in request.files:
            filename = photos.save(request.files['photo'])
            #heli_Test_1.png
            #return filename
            img = open_image(f"static/img/{filename}")
            print(f"static/img/{filename}")
            pred_class,pred_idx,outputs = model.predict(img)
            return json.dumps({
                "predictions": sorted(
                    zip(model.data.classes, map(float, outputs)),
                    key=lambda p: p[1],
                    reverse=True
                )
            })
    return(render_template('main.html'))

if __name__ == '__main__':
    app.run(debug=True)
