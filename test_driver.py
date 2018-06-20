from seg_fwk import segmentation
s = segmentation(imname='twoObj',algo='bhattacharya', dt=0.5)
s.execute()