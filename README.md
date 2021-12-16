# Modified TIDE

## Installation
```shell
pip install git+https://github.com/Swall0w/tide
```

![Confusion Matrix](examples/coco_result.png)

![Per-class information](examples/per_class_info.png)

# Usage
TIDE is meant as a drop-in replacement for the [COCO Evaluation toolkit](https://github.com/cocodataset/cocoapi), and getting started is easy:

```python
from tidecv import TIDE, datasets

tide = TIDE()
tide.evaluate(datasets.COCO(), datasets.COCOResult('path/to/your/results/file'), mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot()       # Show a summary figure. Specify a folder and it'll output a png to that folder.
```

This prints evaluation summary tables to the console:
```
-- mask_rcnn_bbox --

bbox AP @ 50: 61.80

                         Main Errors
=============================================================
  Type      Cls      Loc     Both     Dupe      Bkg     Miss
-------------------------------------------------------------
   dAP     3.40     6.65     1.18     0.19     3.96     7.53
=============================================================

        Special Error
=============================
  Type   FalsePos   FalseNeg
-----------------------------
   dAP      16.28      15.57
=============================
```

And a summary plot for your model's errors:

![A summary plot](https://dbolya.github.io/tide/mask_rcnn_bbox_bbox_summary.png)

## Jupyter Notebook

Check out the [example notebook](https://github.com/dbolya/tide/blob/master/examples/coco_instance_segmentation.ipynb) for more details.


# Datasets
The currently supported datasets are COCO, LVIS, Pascal, and Cityscapes. More details and documentation on how to write your own database drivers coming soon!

# Citation
If you use TIDE in your project, please cite
```
@inproceedings{tide-eccv2020,
  author    = {Daniel Bolya and Sean Foley and James Hays and Judy Hoffman},
  title     = {TIDE: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {ECCV},
  year      = {2020},
}
```

## Contact
For questions about our paper or code, make an issue in this github or contact [Daniel Bolya](mailto:dbolya@gatech.edu). Note that I may not respond to emails, so github issues are your best bet.
