from initialize_phi import phi as ph
from schemes import  chanvesse_functional, yezzi_functional, bhattacharya_functional,seg_w_control,optimal_segmentation
import numpy as np
from PIL import Image
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
            img = np.array(Image.open('airplane.png'))
        elif self.imname == 'zebra':
            img = np.array(Image.open('zebra.bmp'))
        elif self.imname == 'twoObj':
            img = np.array(Image.open('twoObj.bmp'))
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

        elif self.algo == 'ctrl':
            algo_obj = optimal_segmentation(img=self.img, phi_init=self.phi_obj, dt=self.dt)
            algo_obj.run_optimal_segmentation()

        elif self.algo == 'ctrl1':
            algo_obj = seg_w_control(img=self.img, phi_init=self.phi_obj, dt=self.dt)
            algo_obj.run()