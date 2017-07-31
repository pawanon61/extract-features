from tqdm import tqdm
import numpy as np
import sys, os
from skimage import io

img_path = '/braintree/data2/active/common/imagenet_raw/ILSVRC2012_val'

for i in tqdm(range(1000)):
    idx_str = str(i+1)
    filename = os.path.join(img_path, 'ILSVRC2012_val_%s.JPEG'%(idx_str.zfill(8)))
    img = io.imread(filename)  # read mimage as numpy array
    if img.shape == (256,256):
    	img = np.stack((img,)*3, axis=2)
    io.imsave('ILSVRC2012_val_%s.png'%(idx_str.zfill(8)), img)



# # open val.txt to read labels
# f = open('/braintree/data2/active/common/imagenet_raw/val.txt', 'r')
# content = f.readlines()
# f.close()
# list_of_labels = [x.strip() for x in content]
# list_of_labels = list_of_labels[:1000]
# list_of_labels = [s[29:] for s in list_of_labels]
# f = open('labels.txt', 'w')
# for item in list_of_labels:
# 	f.write('%s\n' %str(item))
# f.close()