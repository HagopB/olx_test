# Developping and deploying keras models #

### What is this repository for? 
Implementation of two simple keras models for automatic image quality assessment and product recognition in images (here, watch/clock recognition). The models are further exposed in an API (Flask, deployed on Heroku).

### How do I get set up ?  
Install all requirements. (N.B.: python 3.6.3)
```
pip install -r requirements.txt
```

### How do I see the development process ?
You may follow all development steps and questions in the dev_model.ipynb notebook. (also in .html). However, as the file is quiet big, you would need to clone this repo to visualize the notebook.

### How do I predict on own images ?
you may have information on how to run ```predict.py``` by following:

```
python predict.py --help
```

You just need to store all your images (.jpg, .jpeg, .png) in one unique folder and run ```predict.py``` by following the help instructions. 

### Contents
```
└── olx_test
    ├──  app                         # data folder contaning both A and B images
         ├── models                  # folders where .h5 keras model files are stored
         ├── app.py                  # the flask app
         ├── Procfile                # Procfile necessary for deploying on Heroku
         ├── requirements.txt        # all requirements necessary for the flask app
         └── runtime.txt             # version of python (3.6.3) for deploying on Heroku
    ├── dev_model.html               # The html version of the notebook
    ├── dev_model.ipynb              # The dev notebook
    ├── pedict.py                    # .py to deploy on your own images
    └── utils.py                     # utils (functions call in the notebook)
```
### Acknowledgement
* titu1994 [https://github.com/titu1994/neural-image-assessment](https://github.com/titu1994/neural-image-assessment)
* keras [https://github.com/keras-team/keras](https://github.com/keras-team/keras)



