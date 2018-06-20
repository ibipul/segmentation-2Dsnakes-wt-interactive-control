from initialize_phi import phi as ph
from schemes import  chanvesse_functional
import scipy.ndimage
class segmentation:

    def __init__(self, img, algo,dt):
        self.img = self.get_image()
        self.img_shape = self.img.shape
        self.algo = algo
        self.phi_obj = ph(img=self.img, shape=self.img_shape,mask=None)
        self.dt = dt


    def get_image(self):
        img = scipy.ndimage.imread('airplane.png')#('zebra.bmp')
        return img

    def execute(self):
        algo_obj = chanvesse_functional(img=self.img, phi_init=self.phi_obj, dt=self.dt)
        algo_obj.run_chanvesse()

