#!/usr/bin/env python

def do (pm_clf, pm_ann):
    a = eval_segm.pixel_accuracy(pm_clf,pm_ann) # eval_segm, gt_segm
    print (a)
