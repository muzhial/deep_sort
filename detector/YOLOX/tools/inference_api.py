import os

import numpy as np
import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis


class YOLOXPredictor(object):
    def __init__(
        self,
        exp_name,
        ckpt_file,
        cls_names=COCO_CLASSES,
        use_cuda=True,
        legacy=False,
        is_xywh=False
    ):
        self.exp_name = exp_name
        self.class_names = cls_names
        self.is_xywh = is_xywh
        self.device = 'cuda:0' if use_cuda else 'cpu'

        self.model = self.get_model(ckpt_file)

        self.preproc = ValTransform(legacy=legacy)
        print(f'YOLOX predictor prepared')

    def get_model(self, ckpt_file, conf=0.25, nms=0.45):
        exp = get_exp(None, self.exp_name)
        self.num_classes = exp.num_classes
        self.confthre = conf
        self.nmsthre = nms
        self.test_size = exp.test_size
        model = exp.get_model().to(self.device)
        model.eval()
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        return model

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.class_names)
        return vis_res

    def _xyxy_to_xywh(self, boxes_xyxy):
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xywh = boxes_xyxy.clone()
        elif isinstance(boxes_xyxy, np.ndarray):
            boxes_xywh = boxes_xyxy.copy()

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        return boxes_xywh

    def __call__(self, ori_img, cls_conf=0.35):
        output, img_info = self.inference(ori_img)
        output = output[0]
        ratio = img_info['ratio']
        img = img_info['raw_img']
        if output is None:
            return None, None, None
        output = output.cpu()
        boxes = output[:, 0:4]
        boxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        mask_indice = scores >= cls_conf

        boxes = boxes[mask_indice]
        cls = cls[mask_indice].long()
        scores = scores[mask_indice]

        if boxes.size(0) == 0:
            return None, None, None

        if self.is_xywh:
            boxes = self._xyxy_to_xywh(boxes)

        return boxes.numpy(), scores.numpy(), cls.numpy()

# def imageflow_demo(predictor, vis_folder, current_time, args):
#     cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     save_folder = os.path.join(
#         vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#     )
#     os.makedirs(save_folder, exist_ok=True)
#     if args.demo == "video":
#         save_path = os.path.join(save_folder, args.path.split("/")[-1])
#     else:
#         save_path = os.path.join(save_folder, "camera.mp4")
#     logger.info(f"video save_path is {save_path}")
#     vid_writer = cv2.VideoWriter(
#         save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
#     )
#     while True:
#         ret_val, frame = cap.read()
#         if ret_val:
#             outputs, img_info = predictor.inference(frame)
#             result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
#             if args.save_result:
#                 vid_writer.write(result_frame)
#             ch = cv2.waitKey(1)
#             if ch == 27 or ch == ord("q") or ch == ord("Q"):
#                 break
#         else:
#             break


def main():

    predictor = YOLOXPredictor(
        'yolox-x',
        '/dataset/mz/code/yoloxServer-vehicle/weights/yolox_x.pth')
    # outputs, img_info = predictor.inference('./assets/test.png')
    # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    # cv2.imwrite('./result.png', result_image)

    predictor('./assets/test.png')


if __name__ == "__main__":
    main()
