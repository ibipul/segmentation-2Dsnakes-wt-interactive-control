from scipy.ndimage.morphology import distance_transform_edt as distance
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import numpy as np
#import scipy.io as sio

class phi:
    #mask = sio.loadmat('MASK_zebra.mat')['mask']
    def __init__(self, img, shape, mask):
        self.img = img
        self.phi_shape = shape
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.pix = self.region_select_setup()
        self.selected_region = None
        m = LassoSelector(self.ax1, self.onselect)
        plt.show()
        self.mask = (self.selected_region > 0)*1

    def get_phi(self):
        phi = distance(1 - self.mask) - distance(self.mask) + self.mask
        return phi

    def reset_phi(self, mask):
        phi = distance(1 - self.mask) - distance(self.mask) + self.mask
        return phi

    def region_select_setup(self):
        self.ax1.imshow(self.img)
        x, y = np.meshgrid(np.arange(self.img.shape[1]), np.arange(self.img.shape[0]))
        pix = np.vstack((x.flatten(), y.flatten())).T
        return pix

    def onselect(self, verts):
        p = Path(verts)
        ind = p.contains_points(self.pix, radius=1)
        selected = np.zeros_like(self.img)
        selected.flat[ind] = self.img.flat[ind]
        self.selected_region = selected
        plt.close()




