import tensorflow as tf
import numpy as np
import scipy.misc
from tqdm import tqdm
import sys, os
from sklearn.decomposition import PCA
sys.path.insert(0, '/braintree/home/pgaire/softwares/streams') #to add/import the python module
sys.path.insert(1, '/braintree/home/pgaire/softwares/tools')

import vgg16
import vgg19
import utils

from get_imagenet_images import get_imagenet_images

from streams.envs import hvm
from streams.metrics.neural_cons import NeuralFitAllSites

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
        image = scipy.misc.imresize(image, [224,224])
        output_of_an_image = sess.run(extract_features_from_this_layer, feed_dict={image_placeholder:np.expand_dims(image,0)})
        output_of_an_image = output_of_an_image.flatten()
        features.append(output_of_an_image)
    features = np.asarray(features)
    return features

if feature_extraction_layer == 'pool1':
    tensor_name = 'content_vgg/pool1:0'
elif feature_extraction_layer == 'pool2':
    tensor_name = 'content_vgg/pool2:0'
elif feature_extraction_layer == 'pool3':
    tensor_name = 'content_vgg/pool3:0'
elif feature_extraction_layer == 'pool4':
    tensor_name = 'content_vgg/pool4:0'
elif feature_extraction_layer == 'pool5':
    tensor_name = 'content_vgg/pool5:0'
elif feature_extraction_layer == 'fc6':
    tensor_name = 'content_vgg/Relu:0'
elif feature_extraction_layer == 'fc7':
    tensor_name = 'content_vgg/Relu_1:0'
elif feature_extraction_layer == 'fc8':
    tensor_name = 'content_vgg/fc8/BiasAdd:0'

hvmit = hvm.HvM(var=variation)
list_of_images = hvmit.images #get all images for a variation

image_placeholder = tf.placeholder(dtype=tf.float32, shape=[1,224,224,3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if net == 'vgg16':
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
    	vgg.build(image_placeholder)
elif net == 'vgg19':
    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
    	vgg.build(image_placeholder)

for op in tf.get_default_graph().get_operations():
	print op.name

extract_features_from_this_layer = tf.get_default_graph().get_tensor_by_name(tensor_name)

print ('*******getting var%d images and extracting feature from them*******' %variation)
total_features = extract_features(list_of_images)

if feature_extraction_layer == 'fc6' or feature_extraction_layer == 'fc7' or feature_extraction_layer == 'fc8':
    np.save('/braintree/home/pgaire/data/features_extracted/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), total_features)
else:
    print ("\n******getting 1000 imagenet images and extracting features from them to calculate PCA transform matrix********")
    print ('because the layer you selected has way too many features. so, reducing the features to 1000 per image')
    imagenet_images = get_imagenet_images(nimg = 1000)
    imagenet_features = extract_features(imagenet_images)
    reduced_total_features = PCA(n_components=1000).fit(imagenet_features).transform(total_features)
    np.save('/braintree/home/pgaire/data/features_extracted/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), reduced_total_features)