import tensorflow as tf
from tensorflow.python.platform import gfile
from alexnet import AlexNet
import numpy as np
import scipy.misc
from tqdm import tqdm
import sys, os
sys.path.append('/braintree/home/pgaire/softwares/streams') #to add/import the python module
sys.path.append('/braintree/home/pgaire/softwares/tools')
from get_imagenet_images import get_imagenet_images
from sklearn.decomposition import PCA

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
    # Array for all arguments passed to script
    args = parser.parse_args()
    variation_arg = args.variation
    gpu_arg = args.gpu
    whichlayer_arg = args.whichlayer
    # return all variable values
    return variation_arg, gpu_arg, whichlayer_arg

# match values returned from get_args() to assign to their respective variables
variation, gpu, feature_extraction_layer = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu  # use gpu specified by the user to do the computation

x = tf.placeholder(dtype=tf.float32, shape=[1, 227, 227, 3])
alexnet_model = AlexNet(x, keep_prob=1, num_classes=1000, skip_layer=[],
                        weights_path='DEFAULT')


def extract_features(list_of_images):
    features = []
    for number_of_images, image in enumerate(tqdm(list_of_images)):
        image = scipy.misc.imresize(image, [227,227])
        output_of_an_image = sess.run(extract_feature_from_this_layer, feed_dict={x:np.expand_dims(image,0)})
        output_of_an_image = output_of_an_image.flatten()
        features.append(output_of_an_image)
    features = np.asarray(features)
    return features

sess = tf.Session()
sess.run(tf.global_variables_initializer())
alexnet_model.load_initial_weights(sess)

for op in tf.get_default_graph().get_operations():
    print op.name

if feature_extraction_layer == 'pool1':
    tensor_name = 'pool1:0'
elif feature_extraction_layer == 'pool2':
    tensor_name = 'pool2:0'
elif feature_extraction_layer == 'conv3':
    tensor_name = 'conv3/conv3:0'
elif feature_extraction_layer == 'conv4':
    tensor_name = 'conv4/conv4:0'
elif feature_extraction_layer == 'pool5':
    tensor_name = 'pool5:0'
elif feature_extraction_layer == 'fc6':
    tensor_name = 'fc6/Relu:0'
elif feature_extraction_layer == 'fc7':
    tensor_name = 'fc7/Relu:0'
elif feature_extraction_layer == 'fc8':
    tensor_name = 'fc8/fc8:0'
# print([v.name for v in tf.get_default_graph().get_operations() if 'pool5' in v.name])
# import pdb; pdb.set_trace()
extract_feature_from_this_layer = tf.get_default_graph().get_tensor_by_name(tensor_name)

hvmit = hvm.HvM(var=variation)
list_of_images = hvmit.images #get all the image for a variation
print ('*******getting var%d images and extracting feature from them*******' %variation)
total_features = extract_features(list_of_images)

# if feature_extraction_layer == 'fc6' or feature_extraction_layer == 'fc7' or feature_extraction_layer == 'fc8':
#     np.save('/braintree/home/pgaire/data/features_extracted/alexnet_%s_features_for_var%d_images.npy'%(feature_extraction_layer, variation), total_features)
# else:
#     print ("\n******getting 1000 imagenet images and extracting features from them to calculate PCA transform matrix********")
#     print ('because the layer you selected has way too many features. so, reducing the features to 1000 per image')
#     imagenet_images = get_imagenet_images(nimg = 1000)
#     imagenet_features = extract_features(imagenet_images)
#     reduced_total_features = PCA(n_components=1000).fit(imagenet_features).transform(total_features)
#     np.save('/braintree/home/pgaire/data/features_extracted/alexnet_%s_features_for_var%d_images.npy'%(feature_extraction_layer, variation), reduced_total_features)
print ("\n******getting 1000 imagenet images and extracting features from them to calculate PCA transform matrix********")
print ('because the layer you selected has way too many features. so, reducing the features to 1000 per image')
imagenet_images = get_imagenet_images(nimg = 1000)
imagenet_features = extract_features(imagenet_images)
reduced_total_features = PCA(n_components=1000).fit(imagenet_features).transform(total_features)
np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/alexnet_%s_features_for_var%d_images.npy'%(feature_extraction_layer, variation), reduced_total_features)