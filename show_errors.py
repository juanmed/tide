from tidecv import TIDE, datasets
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', '-a', type=str, default='datasets/coco/annotations/instances_val2017.json')
    parser.add_argument('--result', '-r', default='', type=str)
    parser.add_argument('--name', '-n', default='', type=str)
    parser.add_argument('--show', type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    args = arg()

    tide = TIDE()
    gt = datasets.COCO(args.annotation)
    bbox_results = datasets.COCOResult(args.result)
    #tide.evaluate(datasets.COCO(), datasets.COCOResult('path/to/your/results/file'), mode=TIDE.BOX) # Use TIDE.MASK for masks
    tide.evaluate(gt, bbox_results, mode=TIDE.BOX, name=args.name) # Use TIDE.MASK for masks
    tide.summarize()  # Summarize the results as tables in the console


    ret = tide.get_confusion_matrix()
    cm = pd.DataFrame(data=ret[''].T, 
                      index=gt.classes.values(),
                      columns=gt.classes.values())
    sns.set(font_scale=0.4)
    fig, axes = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, square=True, cbar=True, annot=False, cmap='Blues',
            xticklabels=True, yticklabels=True,
            linewidths=.5
            )
    plt.xlabel("Predict", fontsize=13)
    plt.ylabel("GT", fontsize=13)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig('class_error_confusion_matrix.png')
    #tide.plot() 


if __name__ == '__main__':
    main()
