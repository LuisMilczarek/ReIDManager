#!/usr/bin/env python
import cv2 as cv
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ReIDManager import ReIDManager

MODEL_PATH = "./models/model.pth.tar-20"
YOLO_MODEL = "./yolov8n-pose.pt"
INPUT_FILE = "./sample.mp4"
OUTPUT_FILE = "./output_multi.mp4"
MODEL_NAME = "osnet_x1_0"



def main() -> None:
    cap = cv.VideoCapture(INPUT_FILE)
    fps = cap.get(cv.CAP_PROP_FPS)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    manager = ReIDManager(MODEL_PATH)

    print(f"Video Info:\n\tResolution: {width}x{height}\n\tFPS: {fps:.02f}\n\tTotal Frames: {total_frames}")

    writer = cv.VideoWriter(OUTPUT_FILE,cv.VideoWriter_fourcc(*'mp4v'), fps,(width,height))

    yolo_model = YOLO(YOLO_MODEL)

    for i in range(total_frames):
        print(f"\rFrames: {i:04d}/{total_frames-1:04d}", end="")
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model.track(frame,persist=True,verbose=False)[0]

        if results.boxes.is_track:
            #Coletar todas as boxes, depois inferir

            result : Results
            bboxs = []
            ids = []
            img_patchs = []
            for result in results:
                for (x1,y1,x2,y2), id, score in zip(result.boxes.xyxy.int().cpu().numpy(), result.boxes.id.int().cpu().tolist(), result.boxes.conf.cpu().tolist()):                    
                    if score > 0.75:
                        bboxs.append([x1,y1,x2,y2])
                        ids.append(id)
                        img_patchs.append(frame[y1:y2,x1:x2])
            if len(img_patchs) != 0:                
                ids = manager.extract_ids(ids, img_patchs)
#                print(ids)
                for (x1,y1,x2,y2), id in zip(bboxs, ids):
                        frame = cv.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),3)
                        cv.putText(frame,f"ID: {id}", (x1,y1),cv.FONT_HERSHEY_PLAIN,4, (255,0,0),4)
        writer.write(frame)

    print("\nDone")
    writer.release()
if __name__ == "__main__":
    main()