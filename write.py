import argparse
import sys, os

def get_args():
    # assign description to help doc
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('-m', '--modelname', type=str, help='which model do you want to use', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='which gpu to use in the current hode', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    model_arg = args.modelname
    gpu_arg = args.gpu
	# return all variable values
    return model_arg, gpu_arg
    
model_name, gpu_to_use = get_args()

f = open("/braintree/home/pgaire/data/csv_output/%s_output.csv"%model_name, "w")
f.write("model,var,layer,kind,value\n")
f.close()
if model_name == 'alexnet':
    feature_extraction_layers = ['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7', 'fc8']
elif model_name == 'vgg16' or model_name == 'vgg19':
    feature_extraction_layers = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6', 'fc7', 'fc8']
elif model_name == 'resnet50':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16', 'block17', 'pool', 'fc1000']
elif model_name == 'xception':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'pool', 'fc']
elif model_name == 'mobilenet':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'last_conv']
elif model_name == 'inceptionv1':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11']
elif model_name == 'inceptionv2':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
elif model_name == 'inceptionv3':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13']
elif model_name == 'inceptionv4':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16', 'block17', 'block18', 'block19', 'block20']
elif model_name == 'inceptionresnetv2':
    feature_extraction_layers = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6']
elif model_name == 'splitbrainautoencoder':
    feature_extraction_layers = ['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7']
elif model_name == 'unsupervisedvideo':
    feature_extraction_layers = ['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7']

for variation in range(0,7,3):
    for layer in feature_extraction_layers:
        os.system("python run.py  -m=%s -var=%d -gpu=%d -layer=%s -w=True" %(model_name, variation, gpu_to_use, layer))