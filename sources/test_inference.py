import os
import numpy as np
import json
import torch
import torchvision
import numpy as np
import cv2
import random
import detectron2
import itertools
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

def get_dataset_dicts(img_dir, json_file):
    json_file = os.path.join(img_dir, json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['images']):
        record = {}
        filename = os.path.join(img_dir, v["file_name"])
        
        record["file_name"] = filename
        record["id"] = v["id"]
        record["height"] = v["height"]
        record["width"] = v["width"]
        
        annos = imgs_anns["annotations"]
        objs = []
        for anno in annos:
            if anno['image_id'] == v["id"]:
                # BY pass field
                if anno['category_id'] != 1:
                    cat = 0
                    if anno['category_id'] != 0:
                        cat = anno['category_id'] - 1
                    else:
                        cat = 0
                    # buat polygon
                    xywh = anno['bbox']
                    px = [xywh[0], xywh[0]+xywh[2], xywh[0]+xywh[2], xywh[0]]
                    py = [xywh[1], xywh[1], xywh[1]+xywh[3], xywh[1]+xywh[3]]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = list(itertools.chain.from_iterable(poly))
                    obj = {
                        "bbox": anno['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": [poly],
                        "category_id": cat,
                        "iscrowd": 0
                    }
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def main():
    print("RetinaNet-RoboCup")
    DATASET_PATH = "../RoboCup-Dataset/"
    for d in ["train","valid"]:
        DatasetCatalog.register("robocup_" + d, lambda d=d: get_dataset_dicts(DATASET_PATH + d, d+".json"))
        MetadataCatalog.get("robocup_" + d).set(thing_classes=["ball", "left_goal", "right_goal"])
    robocup_metadata = MetadataCatalog.get("robocup_train")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = "../weights/retinanet_50/"
    cfg.merge_from_file("../../detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ("robocup_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.RETINANET.NUM_CLASSES = 3

    # Cek apakah wights sudah ada
    weights_filename = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if os.path.exists(weights_filename):
        print("Load old weights")
        cfg.MODEL.WEIGHTS = weights_filename
    else:
        print("Download pretrained weights")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_50_FPN_1x/137593951/model_final_b796dc.pkl"  # initialize from model zoo

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()

if __name__ == "__main__":
    main()