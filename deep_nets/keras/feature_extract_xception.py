from keras.applications.xception import Xception
from keras.preprocessing import image_new
from keras.applications.xception import preprocess_input
from keras.models import Model

import tensorflow as tf
import numpy as np
import sys, os
from tqdm import tqdm
import scipy.misc
sys.path.insert(0, '/braintree/home/pgaire/deep_nets/keras') #to add/import the python module
sys.path.insert(1, '/braintree/home/pgaire/softwares/streams') #to add/import the python module
from streams.envs import hvm
from streams.metrics.neural_cons import NeuralFitAllSites
sys.path.insert(2, '/braintree/home/pgaire/softwares/tools')
from get_imagenet_images import get_imagenet_images
from sklearn.decomposition import PCA


import argparse

def get_args():
    # assign description to help doc
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('-variation', '--variation', type=int, help='which variation of image to take', required=True)
    parser.add_argument('-gpu', '--gpu', type=int, help='which gpu to use in the current hode', required=True)
    parser.add_argument('-whichlayer', '--whichlayer', type=str, help='from which layer do you want to extract the feature', required=True)
    parser.add_argument('-model', '--model_name', type=str, help='name of model to run', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    variation_arg = args.variation
    gpu_arg = args.gpu
    whichlayer_arg = args.whichlayer
    model_name_arg = args.model_name
    # return all variable values
    return variation_arg, gpu_arg, whichlayer_arg, model_name_arg

# match values returned from get_args() to assign to their respective variables
variation, gpu, feature_extraction_layer, net = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu #use gpu in the node specified by user


def extract_features(list_of_images):
	features = []
	for number_of_images, image in enumerate(tqdm(list_of_images)):
		# image = scipy.misc.imresize(image, [229,229]) #xception takes image input of size 229x229
		image = image_new.array_to_img(image)
		image = image_new.load_img(image, target_size=(299, 299))
		image = image_new.img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)
		feature_of_this_layer = model.predict(image)
		feature_of_this_layer = feature_of_this_layer.flatten()
		features.append(feature_of_this_layer)
	features = np.asarray(features)
	return features

#after each block after residual features has been added
if feature_extraction_layer == 'block1':
    tensor_name = 'block1_conv2_act'
elif feature_extraction_layer == 'block2':
	tensor_name = 'add_1'
elif feature_extraction_layer == 'block3':
	tensor_name = 'add_2'
elif feature_extraction_layer == 'block4':
	tensor_name = 'add_3'
elif feature_extraction_layer == 'block5':
	tensor_name = 'add_4'
elif feature_extraction_layer == 'block6':
	tensor_name = 'add_5'
elif feature_extraction_layer == 'block7':
	tensor_name = 'add_6'
elif feature_extraction_layer == 'block8':
	tensor_name = 'add_7'
elif feature_extraction_layer == 'block9':
	tensor_name = 'add_8'
elif feature_extraction_layer == 'block10':
	tensor_name = 'add_9'
elif feature_extraction_layer == 'block11':
	tensor_name = 'add_10'
elif feature_extraction_layer == 'block12':
	tensor_name = 'add_11'
elif feature_extraction_layer == 'block13':
	tensor_name = 'add_12'
elif feature_extraction_layer == 'block14':
	tensor_name = 'block14_sepconv2_act'
elif feature_extraction_layer == 'pool':
	tensor_name = 'avg_pool'
elif feature_extraction_layer == 'fc':
	tensor_name = 'fc'

base_model = Xception(weights='imagenet')
# for op in tf.get_default_graph().get_operations():
#     print op.name
model = Model(inputs=base_model.input, outputs=base_model.get_layer(tensor_name).output)

hvmit = hvm.HvM(var=variation)
list_of_images = hvmit.images #get all the image for a variation

print ('*******getting var%d images and extracting feature from them*******' %variation)
total_features = extract_features(list_of_images)

if feature_extraction_layer == 'fc':  # because the output of prediction already has 1000 features, so no need to do pca
    np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), total_features)
else:
    print ("\n******getting 1000 imagenet images and extracting features from them to calculate PCA transform matrix********")
    print ('because the layer you selected has way too many features. so, reducing the features to 1000 per image')
    imagenet_images = get_imagenet_images(nimg = 1000)
    imagenet_features = extract_features(imagenet_images)
    reduced_total_features = PCA(n_components=1000).fit(imagenet_features).transform(total_features)
    np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), reduced_total_features)