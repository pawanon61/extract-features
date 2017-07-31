import sys, os
import numpy as np
sys.path.insert(0, '/braintree/home/pgaire/softwares/streams')
from streams.envs import hvm
from streams.metrics.neural_cons import NeuralFitAllSites
import argparse

def get_args():
    # assign description to help doc
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('-variation', '--variation', type=int, help='which variation of image to take', required=True)
    parser.add_argument('-whichlayer', '--whichlayer', type=str, help='from which layer do you want to extract the feature', required=True)
    parser.add_argument('-model', '--model_name', type=str, help='name of model to run', required=True)
    parser.add_argument('-w', '--write', type=bool, default=False, help='if you want to write the output on file or not. optional argument. pass True if you want to write the output to file')
    # Array for all arguments passed to script
    args = parser.parse_args()
    variation_arg = args.variation
    whichlayer_arg = args.whichlayer
    model_name_arg = args.model_name
    write_arg = args.write
    # return all variable values
    return variation_arg, whichlayer_arg, model_name_arg, write_arg

# match values returned from get_args() to assign to their respective variables
variation, feature_extraction_layer, net, write = get_args()

print ('\nfirst, second and last values are for average over 70-170ms, 100ms and 200ms timepoint respectively')

def calculate_variance(time_point): # to calculate the variance in the given timepoint in ms, if the argument passed is None, the variance will be calculated at the average of time of 70-170 ms
    total_features = np.load('/braintree/home/pgaire/data/features_extracted/features_pretrained/%s_%s_features_for_var%d_images.npy'%(net, feature_extraction_layer, variation))
    hvmit = hvm.HvM(var=variation)
    if variation == 0:  # there are 640 var3 images
        nfit = NeuralFitAllSites(total_features, hvmit.neural(timepoint = time_point), hvmit.meta.obj,
                         n_splits=2, n_splithalves=2)
    else:  # there are 2560 var3 and var6 images
        nfit = NeuralFitAllSites(total_features, hvmit.neural(timepoint = time_point), hvmit.meta.obj,
                         n_splits=2, n_splithalves=2)
    df = nfit.fit()
    variance = df.groupby('site').explained_var.mean().median()
    print ('for var %d images: '%variation, variance)
    return variance

if not write:
    calculate_variance(None)
    calculate_variance(100)
    calculate_variance(200)

if write:
    f = open("/braintree/home/pgaire/data/csv_output/%s_output.csv"%net, "a")
    f.write(net)
    f.write(",")
    f.write(str(variation))
    f.write(",")
    f.write(feature_extraction_layer)
    f.write(",")
    f.write("average,")
    f.write(str(calculate_variance(None)))
    f.write("\n")
    f.write(net)
    f.write(",")
    f.write(str(variation))
    f.write(",")
    f.write(feature_extraction_layer)
    f.write(",")
    f.write("100ms,")
    f.write(str(calculate_variance(100)))
    f.write("\n")
    f.write(net)
    f.write(",")
    f.write(str(variation))
    f.write(",")
    f.write(feature_extraction_layer)
    f.write(",")
    f.write("200ms,")
    f.write(str(calculate_variance(200)))
    f.write("\n")