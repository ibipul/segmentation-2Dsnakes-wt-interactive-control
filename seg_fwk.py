from initialize_phi import phi as ph
from schemes import  chanvesse_functional, yezzi_functional, bhattacharya_functional,seg_w_control,bhattacharya_w_control,optimal_segmentation
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------------------------
# The registry. Adding an image or an algorithm here is all it takes: the browser view (web/) and
# the WebAssembly demo (wasm/) build their pickers from these two tables and load any module this
# file imports, so neither front end needs editing.
# ---------------------------------------------------------------------------------------------

# name passed as `imname` -> image file in the repo root
IMAGES = {
    "twoObj": "twoObj.bmp",
    "zebra": "zebra.bmp",
    "airplane": "airplane.png",
}

# name passed as `algo` -> the class to build and the method that runs it.
#   title: heading in the browser pickers (defaults to the name)
#   label: one-line description in the browser pickers
#   ui:    False keeps it out of the browser pickers (still usable from Python)
ALGOS = {
    "chanvesse": dict(cls=chanvesse_functional, run="run_chanvesse",
                      label="Chan-Vese: separates mean intensity inside vs outside"),
    "yezzi": dict(cls=yezzi_functional, run="run_yezzi",
                  label="Yezzi: mean separation weighted by region area"),
    "bhattacharya": dict(cls=bhattacharya_functional, run="run_bhattacharya",
                         label="Bhattacharyya: separates the intensity histograms"),
    "ctrl": dict(cls=optimal_segmentation, run="run_optimal_segmentation", ui=False),
    "ctrl1": dict(cls=seg_w_control, run="run", title="Control + chanvesse",
                  label="Chan-Vese with interactive feedback control (steer any time; applied every 50 iterations)"),
    "ctrl_bhattacharya": dict(cls=bhattacharya_w_control, run="run", title="Control + Bhattacharya",
                              label="Bhattacharyya with interactive feedback control (steer any time; applied every 50 iterations)"),
}


def ui_options():
    """What the browser pickers show, as plain data: image names and files, algorithm names, titles and labels."""
    return {
        "images": [{"name": n, "file": f} for n, f in IMAGES.items()],
        "algos": [{"name": n, "title": a.get("title", n), "label": a.get("label", "")}
                  for n, a in ALGOS.items() if a.get("ui", True)],
    }


class segmentation:

    def __init__(self, imname, algo,dt):
        self.imname = imname
        self.img = self.get_image()
        self.img_shape = self.img.shape
        self.algo = algo
        self.phi_obj = ph(img=self.img, shape=self.img_shape,mask=None)
        self.dt = dt


    def get_image(self):
        if self.imname not in IMAGES:
            raise ValueError(f"Unknown image {self.imname!r}; choose one of {sorted(IMAGES)}")
        return np.array(Image.open(IMAGES[self.imname]))

    def execute(self):
        if self.algo not in ALGOS:
            raise ValueError(f"Unknown algorithm {self.algo!r}; choose one of {sorted(ALGOS)}")
        spec = ALGOS[self.algo]
        algo_obj = spec["cls"](img=self.img, phi_init=self.phi_obj, dt=self.dt)
        getattr(algo_obj, spec["run"])()
