#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:10:59 2023

@author: fiftak
"""


class ResultPoint:
    "Contains info about one example input and output data"

    def __init__(self, imgs: list, pred: list[int], label: list[int]):
        self.imgs = imgs
        self.pred = pred
        self.label = label


class ResultStorage:
    "Stores ResultPoints and lets one compute metrics"

    def __init__(self):
        self.result_list = []

    def __len__(self):
        return len(self.result_list)

    def add_point(self, resultPoint):
        self.result_list.append(resultPoint)

    def compute_accuracy(self):
        accuracy = 0
        # Iterate over ResultPoints
        for resPt in self.result_list:
            equals = int(resPt.pred == resPt.label)  # 0 or 1
            accuracy += equals
        accuracy /= len(self.result_list)
        return accuracy

    def pred_list(self):
        predList = []
        for resPt in self.result_list:
            predList.append(resPt.pred)
        return predList

    def label_list(self):
        labelList = []
        for resPt in self.result_list:
            labelList.append(resPt.label)
        return labelList
