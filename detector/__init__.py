try:
    from .YOLOv3 import YOLOv3
    has_yolo = True
except (ImportError, ModuleNotFoundError):
    has_yolo = False
try:
    from .MMDet import MMDet
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
from .YOLOX import YOLOXPredictor
has_yolox = True

__all__ = ['build_detector']


def build_detector(cfg, use_cuda):
    if has_mmdet and cfg.USE_MMDET:
        return MMDet(cfg.MMDET.CFG,
                     cfg.MMDET.CHECKPOINT,
                     score_thresh=cfg.MMDET.SCORE_THRESH,
                     is_xywh=True,
                     use_cuda=use_cuda)
    elif has_yolox:
        return YOLOXPredictor(cfg.YOLOX.EXP_NAME,
                              cfg.YOLOX.CKPT_FILE,
                              use_cuda=use_cuda,
                              is_xywh=True)
    elif has_yolo:
        return YOLOv3(cfg.YOLOV3.CFG,
                      cfg.YOLOV3.WEIGHT,
                      cfg.YOLOV3.CLASS_NAMES,
                      score_thresh=cfg.YOLOV3.SCORE_THRESH,
                      nms_thresh=cfg.YOLOV3.NMS_THRESH,
                      is_xywh=True,
                      use_cuda=use_cuda)
