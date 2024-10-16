#!/usr/bin/env python

import os
import torch
import torchreid

import numpy as np

from typing import List

from containers import Tracker

class ReIDManager(object):
    def __init__(self, model_path : str, model_name : str = "resnet50", threshold=0.75, lower_threshold=0.7, img_size=(256,128)) -> None:

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path: {model_path} doenst exist")
        self.__trackers : List[Tracker] = []
        self.__model_path : str = model_path
        self.__model_name = model_name
        self.__threshold = threshold
        self.__lower_threshold = lower_threshold
        self.__trackers_counter = 0
        self.__img_size = img_size
        self.__loadExtractor()

    def __loadExtractor(self):
        self.__extractor = torchreid.utils.FeatureExtractor(
            self.__model_name,
            self.__model_path,
            image_size=self.__img_size
        )

    def extract_id(self, track_id : int, img_patch : np.ndarray) -> int:
        feature = self.__extractor(img_patch)
        for tracker in self.__trackers:
            if track_id == tracker.track_id:
                if self.__lower_threshold < tracker.getDistance(feature) < self.__threshold:
                    tracker.addFeature(feature)
                return tracker.id
        distances = []
        for tracker in self.__trackers:
            distances.append(tracker.getDistance(feature))
        max_dist = 0
        if len(distances) != 0:
            max_dist = max(distances)
        if max_dist > self.__threshold:
            self.__trackers[distances.index(max_dist)].track_id = track_id
            return self.__trackers[distances.index(max_dist)].id
        else:
            self.__trackers.append(Tracker(self.__trackers_counter))
            self.__trackers[-1].addFeature(feature)
            self.__trackers[-1].track_id = track_id
            self.__trackers_counter += 1 
            return self.__trackers[-1].id


