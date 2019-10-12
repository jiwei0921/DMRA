import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.misc import imresize

def imsave(file_name, img, img_size):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: 3*h*w torch tensor
    :return: nothing
    """
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    ndim = len(img.size())
    assert(ndim == 2 or ndim == 3,
           'img must be a 2 or 3 dimensional tensor')

    img = img.numpy()
    img = imresize(img, [img_size[1][0], img_size[0][0]], interp='nearest')
    if ndim == 3:
        plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
    else:
        plt.imsave(file_name, img, cmap='gray')
