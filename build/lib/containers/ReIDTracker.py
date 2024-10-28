#!/usr/bin/env python

import torch

from typing import List

class Tracker(object):
    def __init__(self, id : int, device : str ="cuda") -> None:
        self.__id : int = id
        self.__track_id  : int = id
        self.__device = device
        self.__features = torch.tensor([],device=self.__device)

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
        f = feature
        if len(f.shape) == 1:
            f = torch.unsqueeze(f,0)
        #print(f)
        self.__features = torch.cat((self.__features,f),0)
        return
    
    def getDistance(self, bboxs_features : torch.tensor) -> torch.tensor:
        distances = torch.tensor([],device = self.__device)
        if len(bboxs_features.shape) == 1:
            torch.unsqueeze(bboxs_features,0)
        for bbox_feature in bboxs_features:
            cs = torch.nn.functional.cosine_similarity(bbox_feature, self.__features)
            distances = torch.cat((distances,cs),0)
        if len(distances.shape) == 1:
            distances = torch.unsqueeze(distances,0)
        return torch.max(distances,1)[0]