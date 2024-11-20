#!/usr/bin/env python3

import os
import sys

import numpy as np
import torch
from PIL import Image as Img
import cv2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.inference import load_model


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

GRDINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GR_CKPT = "groundingdino_swint_ogc.pth"

SAM_VERSION = "vit_h"
SAM_CKPT = "sam_vit_h_4b8939.pth"

def load_image_from_array(image_array):
    '''
    Convert numpy array to PIL Image
    '''
    image_pil = Img.fromarray(image_array).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=2000),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # image shape: [3, h, w]
    image, _ = transform(image_pil, None)
    return image_pil, image

def publish_mask_image(masks):
    '''
    Convert numpy array mask to ROS Image message
    '''
    bridge = CvBridge()
    # Regarding only one mask ...
    mask = masks[0]
    mask = mask.cpu().numpy().astype(np.uint8) * 255
    mask = Img.fromarray(mask[0])
    mask = np.array(mask)
    # Convert OpenCV image to ROS Image message
    mask_msg = bridge.cv2_to_imgmsg(mask, encoding="mono8")
    
    print("GROUNDED SAM: SEGMENTATION DONE. Publishing Mask for first time captured...")
    while True:  
        mask_publisher.publish(mask_msg)


class GroundedSAM:
    def __init__(self, target: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load models
        self.grdino = load_model(GRDINO_CONFIG_PATH, GR_CKPT, self.device).to(self.device)
        self.sam = SamPredictor(sam_model_registry[SAM_VERSION](checkpoint=SAM_CKPT).to(self.device))
        # set target object
        self.text_prompt = target if target.endswith(".") else target + "."
        self.text_token = self.grdino.tokenizer(self.text_prompt)
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        self.target_captured = False
        print("GROUNDED SAM: INIT DONE. All models are prepared.")

    def callback(self, msg):
        # convert ROS Image message to OpenCV Image
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # repeat until the target is first captured
        while not self.target_captured:
            self._process(rgb)
            print("No object captured. Retrying ...")


    def _process(self, rgb: np.array):
        image_pil, image = load_image_from_array(rgb)
        image = image.to(self.device)
        # 1. object detection: Ground Dino
        bboxes = self._ground_dino(image)

        # restore from normalized bboxes coordinates to original image's coordinates
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(bboxes.size(0)):
            bboxes[i] = bboxes[i] * torch.Tensor([W, H, W, H])
            bboxes[i][:2] -= bboxes[i][2:] / 2
            bboxes[i][2:] += bboxes[i][:2]
            # if ever succeed in getting target's bbox
            self.target_captured = True

        if not self.target_captured:
            return

        # 2. segmentation: SAM
        self.sam.set_image(image)
        bboxes = self.sam.transform.apply_boxes_torch(bboxes, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=bboxes,
            multimask_output=False,
        )
        # for further process in others ...
        torch.cuda.empty_cache()

        publish_mask_image(masks)

    def _ground_dino(self, image: torch.Tensor):
        with torch.no_grad():
            outputs = self.grdino(image[None], captions=[self.text_prompt])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0] # (nq, 4)
        # filter output
        mask = logits.max(dim=1)[0] > self.box_threshold
        boxes = boxes[mask]     # (num_filtered, 4)

        return boxes
        
if __name__ == "__main__":
    global mask_publisher

    # pre-load model
    target = "lever handle"
    grounded_sam = GroundedSAM(target)
    # ros
    rospy.init_node('realsense_subscriber_gsam', anonymous=True)
    
    mask_publisher = rospy.Publisher("/mask_image", Image, queue_size=10, latch=True)
    rospy.Subscriber("/camera/color/image_raw", Image, grounded_sam.callback)
    rospy.spin()