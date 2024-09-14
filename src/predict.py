# For prediction
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from net.model import UaNet as Net
from utils.util import pad2factor, normalize, crop_boxes2mask
from config import config

class Predictor:
    def __init__(self, checkpoint_path, config=config, threshold=0.72, use_rcnn=False, use_mask=False,
                 target_size=(512, 512)):
        self.config = config
        self.threshold = threshold
        self.use_rcnn = use_rcnn
        self.use_mask = use_mask
        self.target_size = target_size
        self.net = self.load_model(checkpoint_path)

    def read_image(self, image_path, resize=True):
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if resize and self.target_size and len(self.target_size) == 2:
            image_data = cv2.resize(image_data, self.target_size, interpolation=cv2.INTER_AREA)
        return image_data

    @staticmethod
    def image_to_fake_custom_3d(image_data):
        return np.expand_dims(image_data, 0)

    def image_to_model_input(self, image):
        # Preprocess
        if isinstance(image, str) or isinstance(image, os.PathLike):
            image = self.read_image(image)
        original_img = self.image_to_fake_custom_3d(
                            image
                        )
        imgs = original_img.copy()
        imgs = pad2factor(imgs, factor=1)
        imgs = imgs[np.newaxis, ...].astype(np.float32)
        imgs = normalize(imgs)
        imgs = torch.from_numpy(imgs).float()
        return imgs.unsqueeze(0).cuda()

    def predict(self, image_path):
        if not self.net:
            print("Model not loaded!", file=sys.stderr)
            return False
        image = self.read_image(image_path)
        model_input = self.image_to_model_input(image)
        self.net.set_mode('eval')
        self.net.use_rcnn = self.use_rcnn
        self.net.use_mask = self.use_mask
        with torch.no_grad():
            self.net.forward(model_input, None, None, None, None)
        if self.use_mask and self.use_rcnn:
            # Use crop boxes and mask
            pred = self.get_pred_mask(model_input)
        else:
            # Only interpret self.detections
            detections = self.net.detections
            pred = self.visualize_boxes(image, detections)

        return pred

    def load_model(self, checkpoint_path):
        net = Net(self.config).cuda()
        checkpoint = torch.load(checkpoint_path)
        # Load weights into model
        state_dict = net.state_dict()
        state_dict.update({k: v for k, v in checkpoint['state_dict'].items() if k in state_dict})
        net.load_state_dict(state_dict)
        return net


    def get_pred_mask(self, model_input):
        crop_boxes = self.net.crop_boxes
        segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in self.net.mask_probs]
        pred_mask = crop_boxes2mask(crop_boxes[:, 1:], segments, model_input.shape[2:])
        pred_mask = pred_mask.astype(np.uint8)
        return pred_mask

    def visualize_boxes(self, image, boxes, show=True):
        """
        Visualizes bounding boxes on an image based on a confidence threshold.

        Args:
            self: Class instance
            image (np.ndarray): The image to draw on.
            boxes (torch.Tensor): Tensor of shape (N, 8) where:
                                  - boxes[:, 1] is the confidence score
                                  - boxes[:, 3:7] are x, y, width, height of the box
            show (bool): If True, displays the image before returning.
            self.threshold (float): Confidence threshold for displaying boxes (default 0.6).

        Returns:
            np.ndarray: Image with bounding boxes.
        """
        img_copy = image.copy()

        for box in boxes:
            confidence = box[1].item()

            if confidence >= self.threshold:
                x, y, w, h = box[3:7].tolist()
                top_left = (int(x), int(y))
                bottom_right = (int(x + w), int(y + h))

                cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)
                label = f'{confidence:.2f}'
                cv2.putText(img_copy, label, (int(x), int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if show:
            cv2.imshow('Image with Boxes', img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img_copy


def main():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True, type=str)
    parser.add_argument('-i', '--image', required=True, type=str)
    parsed = parser.parse_args()
    # Predict
    checkpoint_path = parsed.checkpoint
    image_path = parsed.image
    predictor = Predictor(checkpoint_path)
    predictor.predict(image_path)

if __name__ == '__main__':
    main()