# extract-features
extract features from deep nets

USAGE EXAMPLE (TO GET THE NEURAL FIT FOR A MODEL, FOR A VARIATION AND FOR A LAYER):

pgaire@braintree-gpu-2:~$ python run.py  -m='alexnet' -var=0 -gpu=3 -layer='pool5'

    -m=MODEL   name of the model you want to use to do the neural fit
    -gpu=ID   which gpu you want to use in the current node
    -var=VARIATION  VARIATION can be 0,3,6 to extract features using var0, var3 and var6 images respectively
    -layer=LAYER    from which layer you want to extract features
     
 
    MODEL can be 'alexnet', 'vgg16', 'vgg19', 'xception', 'mobilenet', 'resnet50', 'inceptionv1', 'inceptionv2', 'inceptionv3', 'inceptionv4', 'inceptionresnetv2', 'splitbrainautoencoder', 'unsupervisedvideo'
     
    VARIATION can be 0, 3, or 6
     
    LAYER can be:
        for alexnet, neural fit can be done to follwing layers and the input to LAYER should be of following format: 
         'conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7', 'fc8'
         
        for vgg16 and vgg19, neural fit can be done to follwing layers and the input to LAYER should be of following format: 
        'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6', 'fc7', 'fc8'
         
        resnet50:
         'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16', 'block17', 'pool', 'fc1000'

        xception:
         'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'pool', 'fc'
         
        mobilenet:
         'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'last_conv'

        incepiton v1:
        'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11'

        inception v2:
        'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12'

        inceptinon v3:
        'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13'

        inceptionv4:
        'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12', 'block13', 'block14', 'block15', 'block16', 'block17', 'block18', 'block19', 'block20'

        inception resnetv2:
        'block1', 'block2', 'block3', 'block4', 'block5', 'block6'

        split brain autoencoder:
        'pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'

        unsupervisedlearning from video:
        'pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'
 
DOCUMENTATION FOR PLOT.PY AND WRITE.PY HOW TO USE:
    to plot use plot.py
    to write the explained variance to file use write.py
    look at argparse funcion for each one if you need to supply values
