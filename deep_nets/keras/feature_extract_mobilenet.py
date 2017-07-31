from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image_new
from keras.applications.mobilenet import preprocess_input
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
		image = image_new.load_img(image, target_size=(224, 224))
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
    tensor_name = 'conv_pw_1_relu'
elif feature_extraction_layer == 'block2':
	tensor_name = 'conv_pw_2_relu'
elif feature_extraction_layer == 'block3':
	tensor_name = 'conv_pw_3_relu'
elif feature_extraction_layer == 'block4':
	tensor_name = 'conv_pw_4_relu'
elif feature_extraction_layer == 'block5':
	tensor_name = 'conv_pw_5_relu'
elif feature_extraction_layer == 'block6':
	tensor_name = 'conv_pw_6_relu'
elif feature_extraction_layer == 'block7':
	tensor_name = 'conv_pw_7_relu'
elif feature_extraction_layer == 'block8':
	tensor_name = 'conv_pw_8_relu'
elif feature_extraction_layer == 'block9':
	tensor_name = 'conv_pw_9_relu'
elif feature_extraction_layer == 'block10':
	tensor_name = 'conv_pw_10_relu'
elif feature_extraction_layer == 'block11':
	tensor_name = 'conv_pw_11_relu'
elif feature_extraction_layer == 'block12':
	tensor_name = 'conv_pw_12_relu'
elif feature_extraction_layer == 'block13':
	tensor_name = 'conv_pw_13_relu'
elif feature_extraction_layer == 'last_conv':
	tensor_name = 'conv_preds'

base_model = MobileNet(weights='imagenet')
# for op in tf.get_default_graph().get_operations():
#     print op.name
model = Model(inputs=base_model.input, outputs=base_model.get_layer(tensor_name).output)

hvmit = hvm.HvM(var=variation)
list_of_images = hvmit.images #get all the image for a variation

print ('*******getting var%d images and extracting feature from them*******' %variation)
total_features = extract_features(list_of_images)

if feature_extraction_layer == 'last_conv':  # because the output of last_conv is already 1000 features
	np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), total_features)
else:
    print ("\n******getting 1000 imagenet images and extracting features from them to calculate PCA transform matrix********")
    print ('because the layer you selected has way too many features. so, reducing the features to 1000 per image')
    imagenet_images = get_imagenet_images(nimg = 1000)
    imagenet_features = extract_features(imagenet_images)
    reduced_total_features = PCA(n_components=1000).fit(imagenet_features).transform(total_features)
    np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), reduced_total_features)
