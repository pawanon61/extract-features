import sys,os
import argparse
#sys.path.insert(0, '/braintree/pgaire/deep_nets')

def get_args():
    # assign description to help doc
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('-var', '--variation', type=int, help='which variation of image to import', required=True)
    parser.add_argument('-m', '--modelname', type=str, help='which model do you want to use', required=True)
    parser.add_argument('-gpu', '--gpu', type=int, help='which gpu to use in the current hode', required=True)
    parser.add_argument('-layer', '--whichlayer', type=str, help='from which layer do you want to extract the feature', required=True)
    parser.add_argument('-w', '--write', type=bool, default=False, help='if you want to write the output on file or not. optional argument. pass True if you want to write the output to file')
    # Array for all arguments passed to script
    args = parser.parse_args()
    var_arg = args.variation
    model_arg = args.modelname
    gpu_arg = args.gpu
    whichlayer_arg = args.whichlayer
    write_arg = args.write
    # return all variable values
    return var_arg, model_arg, gpu_arg, whichlayer_arg, write_arg
    
variation, model_name, gpu_to_use, feature_extract_layer, write = get_args()

if model_name == 'alexnet':
    os.system("python /braintree/home/pgaire/deep_nets/caffe-master/examples/feature_extraction/feature_extract_alexnet.py -gpu=%d -variation=%d -whichlayer=%s" %(gpu_to_use, variation, feature_extract_layer))
elif model_name == 'vgg16' or model_name == 'vgg19':
    os.system("python /braintree/home/pgaire/deep_nets/keras/feature_extract_%s.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(model_name, variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'resnet50':
    os.system("python /braintree/home/pgaire/deep_nets/keras/feature_extract_resnet50.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'xception':
    os.system("python /braintree/home/pgaire/deep_nets/keras/feature_extract_xception.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'mobilenet':
    os.system("python /braintree/home/pgaire/deep_nets/keras/feature_extract_mobilenet.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'inceptionv1':
    os.system("python /braintree/home/pgaire/deep_nets/tf-slim/feature_extract_inception_v1.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'inceptionv2':
    os.system("python /braintree/home/pgaire/deep_nets/tf-slim/feature_extract_inception_v2.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'inceptionv3':
    os.system("python /braintree/home/pgaire/deep_nets/tf-slim/feature_extract_inception_v3.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'inceptionv4':
    os.system("python /braintree/home/pgaire/deep_nets/tf-slim/feature_extract_inception_v4.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'inceptionresnetv2':
    os.system("python /braintree/home/pgaire/deep_nets/tf-slim/feature_extract_inception_resnet_v2.py -variation=%d -gpu=%d -whichlayer=%s -model=%s" %(variation, gpu_to_use, feature_extract_layer, model_name))
elif model_name == 'splitbrainautoencoder':
    os.system("python /braintree/home/pgaire/deep_nets/splitbrainauto-master/feature_extract.py -variation=%d -gpu=%d -whichlayer=%s" %(variation, gpu_to_use, feature_extract_layer))
elif model_name == 'unsupervisedvideo':
    os.system("python /braintree/home/pgaire/deep_nets/unsupervised-video/feature_extract.py -variation=%d -gpu=%d -whichlayer=%s" %(variation, gpu_to_use, feature_extract_layer))

print ('\n*******calculating explained variance*******')
os.system("python /braintree/home/pgaire/softwares/tools/neuralfit.py -variation=%d -whichlayer=%s -model=%s -w=%s" %(variation, feature_extract_layer, model_name, write))