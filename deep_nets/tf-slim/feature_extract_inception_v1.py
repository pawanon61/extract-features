from tqdm import tqdm
import scipy.misc
import numpy as np
import sys, os
sys.path.append('/braintree/home/pgaire/deep_nets/tf-slim/models/slim')
from datasets import imagenet
from nets import inception_v1
from preprocessing import inception_preprocessing
sys.path.append('/braintree/home/pgaire/softwares/streams')
from streams.envs import hvm
sys.path.append('/braintree/home/pgaire/softwares/tools')
from get_imagenet_images import get_imagenet_images
from sklearn.decomposition import PCA
import tensorflow as tf
slim = tf.contrib.slim  #OR import tensorflow.contrib.slim as slim

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

os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu

def extract_features(list_of_images):
    list_of_features = []
    for number_of_images, image in enumerate(tqdm(list_of_images)):
        features = sess.run(output, feed_dict = {x : image})
        features = features.flatten()
        list_of_features.append(features)
    total_features = np.asarray(list_of_features)
    return total_features

if feature_extraction_layer == 'block1':
    tensor_name = 'MaxPool_3a_3x3'
elif feature_extraction_layer == 'block2':
    tensor_name = 'Mixed_3b'
elif feature_extraction_layer == 'block3':
    tensor_name = 'Mixed_3c'
elif feature_extraction_layer == 'block4':
    tensor_name = 'Mixed_4b'
elif feature_extraction_layer == 'block5':
    tensor_name = 'Mixed_4c'
elif feature_extraction_layer == 'block6':
    tensor_name = 'Mixed_4d'
elif feature_extraction_layer == 'block7':
    tensor_name = 'Mixed_4e'
elif feature_extraction_layer == 'block8':
    tensor_name = 'Mixed_4f'
elif feature_extraction_layer == 'block9':
    tensor_name = 'Mixed_5b'
elif feature_extraction_layer == 'block10':
    tensor_name = 'Mixed_5c'
elif feature_extraction_layer == 'block11':
    tensor_name = 'Logits'

checkpoints_dir = '/braintree/home/pgaire/deep_nets/tf-slim/checkpoints/'

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = inception_v1.inception_v1.default_image_size

with tf.Graph().as_default():

    x = tf.placeholder(dtype=tf.float32, shape=[256, 256, 3]) 
    # [256,256,3] because the hvm and imagenet images for pca are of size 256x256

    processed_image = inception_preprocessing.preprocess_image(x, image_size, image_size, is_training=False)
    # the image will be resized to 224x224 and other preprocessing will be applied

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure 
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.

    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, end_points = inception_v1.inception_v1(processed_images, num_classes=1001, is_training=False)

    output = end_points[tensor_name]
    
    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'), slim.get_model_variables('InceptionV1'))
    
    # for op in tf.get_default_graph().get_operations():
    #     print op.name
    # import pdb; pdb.set_trace()

    with tf.Session() as sess:
        
        # Load weights
        init_fn(sess)

        hvmit = hvm.HvM(var=variation)
        hvm_images = hvmit.images #get all images for a variation

        hvm_features = extract_features(hvm_images)

        print ("\n******getting 1000 imagenet images and extracting features from them to calculate PCA transform matrix********")
        print ('because the layer you selected has way too many features. so, reducing the features to 1000 per image')
        imagenet_images = get_imagenet_images(nimg = 1000)

        imagenet_features = extract_features(imagenet_images)

        reduced_total_hvm_features = PCA(n_components=1000).fit(imagenet_features).transform(hvm_features)
        np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation), reduced_total_hvm_features)
       
        
    res = slim.get_model_variables()