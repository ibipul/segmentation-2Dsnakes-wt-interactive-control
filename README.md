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
 
Test Runs:
* Clone the repository
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
