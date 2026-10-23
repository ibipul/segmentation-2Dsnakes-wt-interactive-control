### Segmentation Framework


Descrption:
 * This repository describes a PDE based `segmentation framework by Level set evolution`.
 * Currently `Chanvesse` , `Yezzi`, and `Bhattacharya` energy functionals are implemented, we also provide `interactive feedback-control' implementation.
 * `Freeform initial region selection` is supported. 
 
Project structure:
* `test_driver.py` - contains the basic executable lines.
* `seg_fwk.py` - contains the rudimentary framework with extendable driver & stub sections.
* `initialize_phi.py` - This module helps in distance func initialization with initial region selection.
* `schemes.py` - This module contain all implemented energy functionals, that can be plug and played.
* `redistance.py` - This module contains re-distancing PDE for Level sets. current implementation only has Sussman.
* `run.sh` - one-command launcher for the browser view (see below).
* `web/` - the browser view: `server.py` runs the framework and streams its figures, `index.html` is the page.

Run in the browser:
* ```
    ./run.sh
  ```
* It checks for Python 3.11+, creates `.venv` and installs `requirements.txt` (first run only), finds a free port
  starting at 8090, and opens **http://localhost:8090**. Leave the terminal open; press Ctrl+C there to stop.
  `PORT=9000 ./run.sh` picks the port yourself; `NO_BROWSER=1 ./run.sh` skips opening the browser.
* In the page, pick an image, an algorithm and `dt`, then press Run. The server invokes the framework exactly as in
  Test Runs below. Wherever the framework would open a matplotlib lasso, the page asks you to draw the region on the
  image instead (press, drag, release). With `ctrl1` it asks again every 200 iterations; draw a region to steer the
  contour, or press Skip. The framework's print output shows in the page and in the terminal.
* The framework code is unchanged for this: `web/server.py` replaces `plt.pause`, `plt.show` and `LassoSelector`
  before importing it, so figures are rendered off screen and sent to the page.
 
Test Runs:
* Clone the repository, and install the packages (`./run.sh` does this for you): `pip install -r requirements.txt`
* ```
    from seg_fwk import segmentation
    
    s = segmentation(imname='twoObj',algo='yezzi', dt=0.5)
    s.execute()

    s = segmentation(imname='zebra',algo='bhattacharya', dt=0.5)
    s.execute()
    
    s = segmentation(imname='airplane',algo='chanvesse', dt=0.5)
    s.execute()
    
    s = segmentation(imname='airplane',algo='ctrl1', dt=0.5)
    s.execute()
    
   ```
* Availabe options for imname are: `twoObj`,`zebra`,`airplane`;
  Availabe options for algos are: `chanvesse`,`yezzi`,`bhattacharya`,`ctrl1`.
