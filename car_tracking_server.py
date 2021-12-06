"""
This module gets video in input and outputs the
json file with coordination of bboxes in the video.

"""
import os
import os.path as osp
import warnings
from collections import deque
import copy

from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import argparse
import torch
from deep_sort.sort.iou_matching import iou_cost

from detector import build_detector
from deep_sort import build_tracker
from utils.tools import tik_tok, is_video
from utils.draw import compute_color_for_labels
from utils.parser import get_config
from utils.json_logger import BboxToJsonLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="./demo/ped.avi")
    parser.add_argument("--config_detection",
                        type=str,
                        default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort",
                        type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--write-fps", type=int, default=20)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--cpu",
                        dest="use_cuda",
                        action="store_false",
                        default=True)
    parser.add_argument('--counting_line',
                        action='store_true',
                        default=False)
    parser.add_argument('--deque_maxlen', type=int, default=5)
    parser.add_argument('--c_line_start', nargs='+', default=[470, 230])
    parser.add_argument('--c_line_end', nargs='+', default=[530, 400])
    args = parser.parse_args()

    assert osp.isfile(args.VIDEO_PATH), "Error: Video not found"
    assert is_video(args.VIDEO_PATH), "Error: Not Supported format"
    if args.frame_interval < 1:
        args.frame_interval = 1

    return args


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode!")

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

        # Configure output video and json
        self.logger = BboxToJsonLogger()
        filename, extension = osp.splitext(osp.basename(self.args.VIDEO_PATH))
        self.output_file = osp.join(self.args.save_path, f'{filename}.mp4')
        self.json_output = osp.join(self.args.save_path, f'{filename}.json')
        self.csv_output = osp.join(self.args.save_path, f'{filename}.csv')

        if not osp.isdir(osp.dirname(self.json_output)):
            os.makedirs(osp.dirname(self.json_output))

        if self.args.counting_line:
            self.count_up = {}
            self.count_down = {}
            self.count_up_0 = {}
            self.count_down_0 = {}
            self.count_up_1 = {}
            self.count_down_1 = {}
            self.count_up_0_sum = 0
            self.count_down_0_sum = 0
            self.count_up_1_sum = 0
            self.count_down_1_sum = 0
            self.up_0_identity = set()
            self.down_0_identity = set()
            self.up_1_identity = set()
            self.down_1_identity = set()
            self.pts = {}
            self.pre_count_up = {'car': 0, 'bus': 0, 'truck': 0}
            self.pre_count_down = {'car': 0, 'bus': 0, 'truck': 0}

    def __enter__(self):
        self.vdo.open(self.args.VIDEO_PATH)
        self.total_frames = int(
            cv2.VideoCapture.get(self.vdo, cv2.CAP_PROP_FRAME_COUNT))
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_rate = int(self.vdo.get(cv2.CAP_PROP_FPS))
        self.record_interval = 3 * self.frame_rate
        self.idx_frame = 0

        self.args.c_line_start = [self.im_width // 4, 0]
        self.args.c_line_end = [self.im_width // 4, self.im_height]
        self.args.c1_line_start = [self.im_width // 4 * 3, 0]
        self.args.c1_line_end = [self.im_width // 4 * 3, self.im_height]

        video_details = {'frame_width': self.im_width,
                         'frame_height': self.im_height,
                         'frame_rate': self.args.write_fps,
                         'video_name': self.args.VIDEO_PATH}
        print(f'video info:\n{video_details}')
        codec = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.output_file,
                                      codec,
                                      self.args.write_fps,
                                      (self.im_width, self.im_height))
        self.logger.add_video_details(**video_details)

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def _get_record_per_interval(self, cats=['car', 'bus', 'truck']):
        item = []
        item.append(self.idx_frame)
        for cat in cats:
            if self.count_up.get(cat):
                item.append(
                    self.count_up.get(cat) - self.pre_count_up.get(cat)
                )
                self.pre_count_up[cat] = self.count_up[cat]
            else:
                item.append(0)

            if self.count_down.get(cat):
                item.append(
                    self.count_down.get(cat) - self.pre_count_down.get(cat)
                )
                self.pre_count_down[cat] = self.count_down[cat]
            else:
                item.append(0)
        return item

    def run(self):
        record_name = ['frame_id',
                       'car_up',
                       'car_down',
                       'bus_up',
                       'bus_donw',
                       'truck_up',
                       'truck_down']
        record_items = []
        idx_frame = 0
        pbar = tqdm(total=self.total_frames + 1)
        while self.vdo.grab():
            if idx_frame % args.frame_interval == 0:
                _, ori_im = self.vdo.retrieve()
                timestamp = self.vdo.get(cv2.CAP_PROP_POS_MSEC)
                frame_id = int(self.vdo.get(cv2.CAP_PROP_POS_FRAMES))
                self.logger.add_frame(frame_id=frame_id, timestamp=timestamp)
                self.detection(frame=ori_im, frame_id=frame_id)
                self.save_frame(ori_im)
                self.idx_frame += 1
                # TODO: is ugly, to optimize later
                if self.idx_frame % self.record_interval == 0:
                    item = self._get_record_per_interval()
                    record_items.append(item)
            idx_frame += 1
            pbar.update()
        self.logger.json_output(self.json_output)
        res_csv = pd.DataFrame(columns=record_name, data=record_items)
        res_csv.to_csv(self.csv_output, encoding='utf-8')

    # @tik_tok
    def detection(self, frame, frame_id):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # do detection
        # bbox_xywh, cls_conf, cls_ids = self.detector(im)  # not YOLOX
        bbox_xywh, cls_conf, cls_ids = self.detector(frame)  # YOLOX
        if bbox_xywh is not None:
            # select person class
            # mask = cls_ids == 0
            # select car|bus|truck
            mask = np.where(
                (cls_ids == 2) | (cls_ids == 5) | (cls_ids == 7))[0]

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
            cls_conf = cls_conf[mask]
            cls_id = cls_ids[mask]
            assert (
                len(cls_conf) == len(cls_id) and len(cls_conf) == len(bbox_xywh)
            ), 'cls conf, output length not equal'

            # draw boxes without tracking
            # self.draw_boxes_no_track(img=frame,
            #                          frame_id=frame_id,
            #                          boxes=bbox_xywh,
            #                          cls_ids=cls_ids)
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im, cls_id)

            # draw boxes for visualization
            if len(outputs) > 0:
                frame = self.draw_boxes(
                    img=frame, frame_id=frame_id, output=outputs)
        # cv2.line(frame,
        #          self.args.c_line_start, self.args.c_line_end,
        #          (0, 255, 0), 2)
        # cv2.line(frame,
        #          self.args.c1_line_start, self.args.c1_line_end,
        #          (0, 255, 0), 2)

    def draw_boxes(
        self,
        img,
        frame_id,
        output,
        offset=(0, 0)
    ):
        for i, box in enumerate(output):
            x1, y1, x2, y2, identity, class_id = [int(ii) for ii in box]
            self.logger.add_bbox_to_frame(frame_id=frame_id,
                                          bbox_id=identity,
                                          top=y1,
                                          left=x1,
                                          width=x2 - x1,
                                          height=y2 - y1)
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # box text and bar
            category = self.detector.class_names[class_id]
            self.logger.add_label_to_bbox(frame_id=frame_id,
                                          bbox_id=identity,
                                          category=category,
                                          confidence=0.73)
            color = compute_color_for_labels(identity)
            label = '{} {:d}'.format(category, identity)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(img,
                          (x1, y1),
                          (x1 + t_size[0], y1 + t_size[1]),
                          color, -1)
            cv2.putText(img, label,
                        (x1, y1 + t_size[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)

            if self.args.counting_line:
                self.pts.setdefault(
                    identity,
                    deque(maxlen=self.args.deque_maxlen))
                box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2),
                              x1, y1, x2, y2)

                if len(self.pts.get(identity, [])) != 0:
                    iou_status = self.iou_filter(
                        self.pts.get(identity)[-1][2:], box_center[2:])
                    if iou_status:
                        self.pts[identity].append(box_center)
                else:
                    self.pts[identity].append(box_center)

                if len(self.pts[identity]) >= 2:
                    p0 = self.pts[identity][-2][:2]
                    p1 = self.pts[identity][-1][:2]
                    if self.intersect(p0, p1,
                                      self.args.c1_line_start,
                                      self.args.c1_line_end):
                        if p1[0] > p0[0] and identity not in self.down_1_identity:
                            self.count_down_1.setdefault(category, 0)
                            self.count_down_1[category] += 1
                            self.down_1_identity.add(identity)
                            self.count_down_1_sum += 1
                        elif p1[0] <= p0[0] and identity not in self.up_1_identity:
                            self.count_up_1.setdefault(category, 0)
                            self.count_up_1[category] += 1
                            self.up_1_identity.add(identity)
                            self.count_up_1_sum += 1
                    if self.intersect(p0, p1,
                                      self.args.c_line_start,
                                      self.args.c_line_end):
                        if p1[0] > p0[0] and identity not in self.down_0_identity:
                            self.count_down_0.setdefault(category, 0)
                            self.count_down_0[category] += 1
                            self.down_0_identity.add(identity)
                            self.count_down_0_sum += 1
                        elif p1[0] <= p0[0] and identity not in self.up_0_identity:
                            self.count_up_0.setdefault(category, 0)
                            self.count_up_0[category] += 1
                            self.up_0_identity.add(identity)
                            self.count_up_0_sum += 1
        for cat in ['car', 'bus', 'truck']:
            self.count_down[cat] = max(self.count_down_0.get(cat, 0),
                                       self.count_down_1.get(cat, 0))
            self.count_up[cat] = max(self.count_up_0.get(cat, 0),
                                     self.count_up_1.get(cat, 0))
        # count_down = sum(self.count_down.values())
        # count_up = sum(self.count_up.values())

        count_down = max(self.count_down_0_sum, self.count_down_1_sum)
        count_up = max(self.count_up_0_sum, self.count_up_1_sum)
        cv2.putText(img,
                    f'total: {count_up + count_down}',
                    (self.im_width // 3, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)
        cv2.putText(img,
                    f'R: {count_down}',
                    (self.im_width // 3, 75),
                    cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)
        cv2.putText(img,
                    f'L: {count_up}',
                    (self.im_width // 3, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)
        # cv2.putText(img,
        #             f"car: {self.count_down['car'] + self.count_up['car']}",
        #             (self.im_width // 3 + 150, 50),
        #             cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)
        # cv2.putText(img,
        #             f"bus: {self.count_down['bus'] + self.count_up['bus']}",
        #             (self.im_width // 3 + 150, 75),
        #             cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)
        # cv2.putText(img,
        #             f"truck: {self.count_down['truck'] + self.count_up['truck']}",
        #             (self.im_width // 3 + 150, 100),
        #             cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)

        return img

    def iou_filter(self, box1, box2):
        lx = max(box1[0], box2[0])
        ly = max(box1[1], box2[1])
        rx = min(box1[2], box2[2])
        ry = min(box1[3], box2[3])
        if rx <= lx or ry <= ly:
            return False
        else:
            return True


    def save_frame(self, frame) -> None:
        if frame is not None:
            self.writer.write(frame)

    def draw_boxes_no_track(self, img, frame_id, boxes, cls_ids):
        for i, box in enumerate(boxes):
            box[:2] = box[:2] - box[2:] / 2
            box[2:] = box[:2] + box[2:]
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            cls_id = 0
            color = compute_color_for_labels(cls_id)
            label = '{}{:d}'.format("", cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        return img

    @classmethod
    def ccw(cls, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @classmethod
    def intersect(cls, A, B, C, D):
        return (
            cls.ccw(A, C, D) != cls.ccw(B, C, D) and
            cls.ccw(A, B, C) != cls.ccw(A, B, D))


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    # print(f'cfg:\n{cfg}')

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
