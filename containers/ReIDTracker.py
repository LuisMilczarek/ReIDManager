#!/usr/bin/env python

import torch

from typing import List

class Tracker(object):
    def __init__(self, id : int) -> None:
        self.__id = id
        self.__track_id = id
        self.__features : List[torch.tensor] = []

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def track_id(self) -> int:
        return self.__track_id

    @track_id.setter
    def track_id(self, id) -> None:
        self.__track_id = id

    def addFeature(self, feature : torch.tensor) -> None:
        self.__features.append(feature)
        return
    
    def getDistance(self, feature : torch.tensor) -> float:
        distances = []
        for feature_i in self.__features:
            distances.append(torch.nn.functional.cosine_similarity(feature, feature_i).item())
        return max(distances)