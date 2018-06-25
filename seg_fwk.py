from initialize_phi import phi as ph
from schemes import  chanvesse_functional, yezzi_functional, bhattacharya_functional,seg_w_control
import scipy.ndimage
class segmentation:

    def __init__(self, imname, algo,dt):
        self.imname = imname
        self.img = self.get_image()
        self.img_shape = self.img.shape
        self.algo = algo
        self.phi_obj = ph(img=self.img, shape=self.img_shape,mask=None)
        self.dt = dt


    def get_image(self):
        if self.imname == 'airplane':
            img = scipy.ndimage.imread('airplane.png')
        elif self.imname == 'zebra':
            img = scipy.ndimage.imread('zebra.bmp')
        elif self.imname == 'twoObj':
            img = scipy.ndimage.imread('twoObj.bmp')
        return img

    def execute(self):
        if self.algo == 'chanvesse':
            algo_obj = chanvesse_functional(img=self.img, phi_init=self.phi_obj, dt=self.dt)
            algo_obj.run_chanvesse()

        elif self.algo == 'yezzi':
            algo_obj = yezzi_functional(img=self.img, phi_init=self.phi_obj, dt=self.dt)
            algo_obj.run_yezzi()

        elif self.algo == 'bhattacharya':
            algo_obj = bhattacharya_functional(img=self.img, phi_init=self.phi_obj, dt=self.dt)
            algo_obj.run_bhattacharya()

        elif self.algo == 'ctrl1':
            algo_obj = seg_w_control(img=self.img, phi_init=self.phi_obj, dt=self.dt)
            algo_obj.run()