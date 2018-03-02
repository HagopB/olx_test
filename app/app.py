from flask import Flask, render_template, request, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
from keras.models import Model
from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Dense, Dropout
from PIL import Image
import tensorflow as tf
from io import BytesIO
import requests

# init app
app = Flask(__name__)
model_1 = None
model_2 = None

###########################################################
##################### Util functions ######################

def synsets2clocks(probas):
    """ Allows to pass from the 1000 synset probas to custom case
        Simple summing of labels belonging to watch/clock children synsets
    """
    clock_ids = [409, 530, 531, 826, 892]
    probas_clocks_ = probas[0, clock_ids].sum()
    return probas_clocks_

def mean_score(scores):
    """computes the IQA mean score
       Please refer to NIMA article for further precisions
    """
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

###########################################################
##################### Loading models ######################
def IQA():
    """ getting the NIMA IQA pre-trained model """
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)

    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('./models/inception_resnet_weights.h5')
    return model

def WatchClassifier():
    """Loads pre-trained model for watch-classification"""
    model = ResNet50(include_top=True, weights=None)
    model.load_weights('./models/weights_resnet50.h5')
    return model

def load_models():
    """Loading all models"""
    global model_1, model_2
    model_1 = IQA()
    model_2 = WatchClassifier()
    global graph_1, graph_2
    graph_1 = tf.get_default_graph()
    graph_2 = tf.get_default_graph()

###########################################################
######################### Predict #########################

def predict_IQA(img):
    """Predict the IQA"""
    img = 2*(img/255.0)-1.0 # process images
    with graph_1.as_default():
        scores = model_1.predict(img) # predict
        mean_score_ = mean_score(scores)
        return mean_score_

def predict_WatchClassifier(img, th=0.01):
    """Predict if there is a watch in the image"""
    with graph_2.as_default():
        _probas = model_2.predict(img)
        probas_ = synsets2clocks(_probas)
        labels_ = probas_ > th
        return labels_, probas_

###########################################################
###################### API endpoints ######################

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    # reading image
    if 'image' in request.files:
        image = request.files['image']
        image = Image.open(image)
    elif 'url' in request.json:
        url = request.json['url']
        resp = requests.get(url)
        image = Image.open(BytesIO(resp.content))

    image = np.asarray(image.resize((224, 224), Image.ANTIALIAS))
    image = image.reshape(1, 224, 224, 3)

    # check if it is still loaded, Heroku issue
    if (model_1 is None) or (model_2 is None):
        load_models()

    # predict
    iqa = predict_IQA(image)
    labels, probas = predict_WatchClassifier(image)
    return jsonify({'watch_pred': (str(labels), str(probas)), 'quality_score':str(iqa)})

# Model status check
@app.route('/status', methods=['GET'])
def status():
    if (model_1 is None) or (model_2 is None):
        return 'Model not loaded'
    else:
        return 'Model ready for prediction'

if __name__ == '__main__':
    print('Loading keras models and starting server')
    load_models()
    app.run(debug=True)
