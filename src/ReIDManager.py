#!/usr/bin/env python

import os
import torch
import torchreid

import numpy as np

from typing import List

from containers import Tracker

class ReIDManager(object):
    def __init__(self, model_path : str, model_name : str = "resnet50", threshold=0.73, lower_threshold=0.63, img_size=(256,128), device : str = "cuda") -> None:

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path: {model_path} doenst exist")
        self.__trackers : List[Tracker] = []
        self.__model_path : str = model_path
        self.__model_name = model_name
        self.__threshold = threshold
        self.__lower_threshold = lower_threshold
        self.__trackers_counter = 0
        self.__img_size = img_size
        self.__device = device
        self.__loadExtractor()

    def __loadExtractor(self):
        self.__extractor = torchreid.utils.FeatureExtractor(
            self.__model_name,
            self.__model_path,
            image_size=self.__img_size,
        )

    def extract_id(self, track_id : int, img_patch : np.ndarray) -> int:
        feature = self.__extractor(img_patch)
        for tracker in self.__trackers:
            if track_id == tracker.track_id:
                if self.__lower_threshold < tracker.getDistance(feature) < self.__threshold:
                    tracker.addFeature(feature)
                return tracker.id
        distances = torch.tensor([],device=self.__device)
        for tracker in self.__trackers:
            distances = torch.cat((distances,tracker.getDistance(feature)),0)
        max_dist = 0
        if len(distances) != 0:
            max_dist = torch.max(distances).cpu().item()
        if max_dist > self.__threshold:
            print(distances)
            self.__trackers[torch.argmax(distances)].track_id = track_id
            return self.__trackers[torch.argmax(distances)].id
        else:
            self.__trackers.append(Tracker(self.__trackers_counter))
            self.__trackers[-1].addFeature(feature)
            self.__trackers[-1].track_id = track_id
            self.__trackers_counter += 1 
            return self.__trackers[-1].id
    
   
    def extract_ids(self, track_ids : List[int], img_patchs : List[np.ndarray]) -> List[int]:
        if len(track_ids) != len(img_patchs):
            raise ValueError(f"Track_ids (size {len(track_ids)} size is different from img_patchs (size {len(img_patchs)})")
        ids = []
        length = len(track_ids)
        features = self.__extractor(img_patchs)
        for i in range(length):
            id = -1
            for tracker in self.__trackers:
                if track_ids[i] == tracker.track_id:
                    id = tracker.id
                    if self.__lower_threshold < tracker.getDistance(torch.unsqueeze(features[i],0)) < self.__threshold:
                        tracker.addFeature(features[i])
                    break
            ids.append(id)
        
        if not -1 in ids:
            return ids
        
        if len(self.__trackers) != 0:
            not_selected_track_ids_indexes = np.where(np.array(ids) == -1)[0]
            not_selected_features = features[not_selected_track_ids_indexes]

            trackers_not_matched_indexes = []
            for i, tracker in enumerate(self.__trackers):
                if tracker.id not in ids:
                    trackers_not_matched_indexes.append(i)


            distance_matrix = torch.tensor([], dtype=torch.float64, device=self.__device)
            for i in trackers_not_matched_indexes:
                distance_matrix = torch.cat((distance_matrix, torch.unsqueeze(self.__trackers[i].getDistance(not_selected_features),0)),0)

            # print(distance_matrix)

            if distance_matrix.numel() > 0:
            
                index_max = torch.argmax(distance_matrix)
                distance_matrix_height = len(distance_matrix.shape)
                distance_matrix_width = distance_matrix.shape[0]
    
                # print(distance_matrix)
                # print(index_max)
    
    
                while distance_matrix[index_max % distance_matrix_width, index_max // distance_matrix_width, ] > self.__threshold:
                    ids[not_selected_track_ids_indexes[index_max // distance_matrix_width]] = self.__trackers[trackers_not_matched_indexes[index_max % distance_matrix_width]].id
    
                    distance_matrix[index_max % distance_matrix_width,:] = -1
                    distance_matrix[:, index_max // distance_matrix_width] = -1
    
                    index_max = torch.argmax(distance_matrix)
                    # print(distance_matrix)
                    # print(index_max % distance_matrix_width)
        for i in range(length):
            if ids[i] == -1:
                self.__trackers.append(Tracker(self.__trackers_counter))
                self.__trackers[-1].addFeature(torch.unsqueeze(features[i],0))
                self.__trackers[-1].track_id = track_ids[i]
                self.__trackers_counter += 1 
                ids[i] = self.__trackers[-1].id
        return ids