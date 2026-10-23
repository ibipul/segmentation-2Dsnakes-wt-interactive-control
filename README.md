### Segmentation Framework

Quick start (runs locally, view it in your browser):
* ```
    git clone https://github.com/ibipul/segmentation-2Dsnakes-wt-interactive-control.git
    cd segmentation-2Dsnakes-wt-interactive-control
    ./run.sh
  ```
  Needs Python 3.11+. The first run installs the packages into `.venv`; then the page opens at
  **http://localhost:8090** (or the next free port; the terminal prints the address).
* In the page: pick an image, an algorithm and `dt`, press Run, then draw the starting region on the image
  (press, drag, release). The contour evolves live.
* With **Control + chanvesse** (`ctrl1`) or **Control + Bhattacharya** (`ctrl_bhattacharya`) you can steer the
  contour while it runs: draw a region around an area that should be inside the object, any time. It's applied at the
  next checkpoint (every 50 iterations in the page, every 200 when run from Python); until then it shows as a
  dashed yellow outline.
* To run without the browser, see Test Runs below.

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
* `run.sh` - one-command launcher for the browser view (see Quick start).
* `web/` - the browser view: `server.py` runs the framework and streams its figures, `index.html` is the page.

More on the browser view:
* `./run.sh` checks for Python 3.11+, creates `.venv` and installs `requirements.txt` (first run only), finds a free
  port starting at 8090, and opens the page. Leave the terminal open; press Ctrl+C there to stop.
  `PORT=9000 ./run.sh` picks the port yourself; `NO_BROWSER=1 ./run.sh` skips opening the browser.
* The server invokes the framework exactly as in Test Runs below, one run at a time. The framework's print output
  shows in the page and in the terminal.
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

    s = segmentation(imname='zebra',algo='ctrl_bhattacharya', dt=0.5)
    s.execute()
    
   ```
* Availabe options for imname are: `twoObj`,`zebra`,`airplane`;
  Availabe options for algos are: `chanvesse`,`yezzi`,`bhattacharya`,`ctrl1`,`ctrl_bhattacharya`.
