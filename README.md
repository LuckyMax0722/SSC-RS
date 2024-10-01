# SSC-RS

## SSC-RS: Elevate LiDAR Semantic Scene Completion with Representation Separation and BEV Fusion

This repository implement the inference script for SSA-SC. For original code, please refer to [SSC-RS](https://github.com/Jieqianyu/SSC-RS)

## Getting Start
Clone the repository:
```
git clone https://github.com/LuckyMax0722/SSC-RS.git
```

### Inference

You can use the `infer_dsec.py` file to infer the sematic voxel output. 

Also, you need to set the path to the pre-trained model and the dataset root directory.


```
$ cd <root dir of this repo>
$ python infer_dsec.py
```

### Pretrained Model

You can download the models with the scores below from this [Google drive link](https://drive.google.com/file/d/1-b3O7QS6hBQIGFTO-7qSG7Zb9kbQuxdO/view?usp=sharing),

| Model  | Segmentation | Completion |
|--|--|--|
| SSC-RS | 24.2 | 59.7 |