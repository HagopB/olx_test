# utf-8
from optparse import OptionParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#############################################################

# option OptionParser

parser = OptionParser()

parser.add_option('--path_to_images',
                dest='path_to_images',
                help='The Path to test images. It can be .png, .jpeg, .jpg')

parser.add_option('--path_output_file',
                dest='path_output_file',
                default='./results.pkl',
                help='Please indicated the path where the output will be saved. Note that it should be a pickle')

(options, args) = parser.parse_args()

#############################################################

if not options.path_to_images: # if empty
    parser.error('Error: The path to test images is empty. Pass --path_to_images to command line')

from utils import *
import pickle

path_to_images = options.path_to_images
path_output_file = options.path_output_file

print('Path to images: {}'.format(path_to_images))
print('Path to output file: {}'.format(path_output_file))

##############################################################

# LOADING IMAGES
print('Loading images')
test_images, test_index =  load_images(path_to_images)
print('Found', len(test_images), 'images to predict on')

# LOADING MODELS
print('Loading models')
iqa = IQA('./app/models/inception_resnet_weights.h5')
tfl = ResNet50(include_top=True, weights='./app/models/weights_resnet50.h5')

# PREDICT
print('Predicting probas & labels: is there a watch ?')
th = 0.01
_probas = tfl.predict(test_images, batch_size=1, verbose=True)
probas_ = synsets2clocks(_probas)
labels_ = [el > th for el in probas_]

print('Predicting Image Quality scores')
#preprocessing adapted to inceptionresnet_v2
test_images = 2*(test_images/255.0)-1.0

# mean scores
scores = iqa.predict(test_images, batch_size=1, verbose=1)
scores_ = [mean_score(el) for el in scores]

out_dict = dict()
for idx, el in enumerate(test_index):
    out_dict[el] = dict()
    out_dict[el]['quality_score'] = scores_[idx]
    out_dict[el]['watch_pred'] = (labels_[idx], probas_[idx])

# SAVING JSON FILE
pickle.dump(out_dict, open(path_output_file, 'wb'))
