# update

- YOLOX --> detector
- Baidu AICityReID --> extractor

# run

## checkpoint

### deep

```shell
mv ckpt.t7 deep_sort/deep/checkpoint
```

### AICityReID

```shell
mv net_last.pth extractor/AICityReID/checkpoints/
```

### yolox

配置在 `configs/deep_sort.ymal`。

## 运行

```shell
bash run.sh
```


# Training the RE-ID model
The original model used in paper is in original_model.py, and its parameter here [original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).

To train the model, first you need download [Market1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset or [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) dataset.

Then you can try [train.py](deep_sort/deep/train.py) to train your own parameter and evaluate it using [test.py](deep_sort/deep/test.py) and [evaluate.py](deep_sort/deep/evalute.py).
![train.jpg](deep_sort/deep/train.jpg)


# References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
