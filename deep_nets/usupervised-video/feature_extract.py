import sys, os
import numpy as np
sys.path.append('/braintree/home/pgaire/softwares/streams')
from streams.envs import hvm
import caffe
import scipy.misc
from tqdm import tqdm
sys.path.append('/braintree/home/pgaire/softwares/tools')
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
    # Array for all arguments passed to script
    args = parser.parse_args()
    variation_arg = args.variation
    gpu_arg = args.gpu
    whichlayer_arg = args.whichlayer
    # return all variable values
    return variation_arg, gpu_arg, whichlayer_arg

# match values returned from get_args() to assign to their respective variables
variation, gpu, feature_extraction_layer = get_args()

if feature_extraction_layer == 'pool1':
    layer = 'pool1'
elif feature_extraction_layer == 'pool2':
    layer = 'pool2'
elif feature_extraction_layer == 'conv3':
    layer = 'caffe.SpatialConvolution_8'
elif feature_extraction_layer == 'conv4':
    layer = 'caffe.SpatialConvolution_10'
elif feature_extraction_layer == 'pool5':
    layer = 'pool5'
elif feature_extraction_layer == 'fc6':
    layer = 'caffe.InnerProduct_16'
elif feature_extraction_layer == 'fc7':
    layer = 'caffe.InnerProduct_19'

caffe.set_device(gpu)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

def extract_features(list_of_images):
    total_features = []
    for number_of_images, image in enumerate(tqdm(list_of_images)):
        # image = scipy.misc.imresize(image, [227,227])
        transformed_image = transformer.preprocess('data_caffe', image)
        # copy the image data into the memory allocated for the net
        net.blobs['data_caffe'].data[...] = transformed_image
        # output = net.forward(start='conv1', end='fc8')
        output = net.forward()
        features = net.blobs[layer].data[0]
        features = features.flatten()
        total_features.append(features)
    total_features = np.asarray(total_features)
    return total_features

caffe_root = '/braintree/home/pgaire/deep_nets/caffe-master/'

model_root = '/braintree/home/pgaire/deep_nets/unsupervised-video/'
model_file = model_root + 'models/caffemodels/sup_coco.caffemodel'

deploy_prototxt = model_root + 'models/caffemodels/deploy.prototxt'

net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
# net = caffe.Net(deploy_prototxt, caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data_caffe': net.blobs['data_caffe'].data.shape})
transformer.set_transpose('data_caffe', (2,0,1))  # move image channels to outermost dimension
# transformer.set_mean('data_caffe', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data_caffe', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data_caffe', (2,1,0))  # swap channels from RGB to BGR


if layer not in net.blobs:
    raise TypeError("Invalid layer name: " + layer)

net.blobs['data_caffe'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
hvmit = hvm.HvM(var=variation)
list_of_images = hvmit.images
hvm_features = extract_features(list_of_images)

imagenet_images = get_imagenet_images(nimg = 1000)
imagenet_features = extract_features(imagenet_images)

reduced_total_features = PCA(n_components=1000).fit(imagenet_features).transform(hvm_features)

# np.save('features.npy', reduced_total_features)

np.save('/braintree/home/pgaire/data/features_extracted/features_pretrained/unsupervisedvideo_%s_features_for_var%d_images.npy'%(feature_extraction_layer, variation), reduced_total_features)