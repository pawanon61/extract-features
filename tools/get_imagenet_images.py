import numpy as np
import h5py
 
def get_imagenet_images(nimg):
    n_img_per_class = (nimg - 1) // 1000
    base_idx = np.arange(n_img_per_class).astype(int)
    idx = []
    for i in range(1000):
        idx.extend(50 * i + base_idx)

    for i in range((nimg - 1) % 1000 + 1):
        idx.extend(50 * i + np.array([n_img_per_class]).astype(int))

    with h5py.File('/braintree/home/pgaire/data/imagenet2012.hdf5') as f:
        ims = np.array([f['val/images'][i] for i in idx])/255.
    return ims