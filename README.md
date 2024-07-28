# Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization
This repository contains code for our TMM paper ["Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization"](https://ieeexplore.ieee.org/abstract/document/10306334).

# Dependencies
We listed the dependencies in the `requirements.txt`.

# Data and pre-trained models
We used the Microsoft COCO dataset to pre-train the mentioned styler transfer network and the CASIA-WebFace dataset to pre-train the face regconition networks. We used LFW, VGGFace2, AgeDB and CFP as the tested datasets. The pre-trained face recognition models and styler transfer models can be found [here](https://drive.google.com/drive/folders/1eXhHl7YnBgUuHv473_-qDodd9f3IogYa?usp=sharing). 

# Usage
Examples of training and evaluation scripts can be found in:
```
python train_stymask.py
```
and 
```
python test_stymask.py
```
The default arguements are set in `config.py`. If need to try other parameters, please modify it directly.

ps: 1. download the pretrained models via the above link and create a new folder `checkpoint` in the project, and inside the `checkpoint`, create two folders, i.e., `face_models` and `style_models`. Put the face model into the `face_models` folder and the style model into the `style_models` folder.
2. download the files in `data` folder, create a `data` folder in the project and unzip the lfw data in the `data` folder. The lfw dataset are from the lfw official website. 

# Reference
If you find our paper/this repo useful for your research, please consider citing our work:
```
@article{gong2023stealthy,
  title={Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization},
  author={Gong, Huihui and Dong, Minjing and Ma, Siqi and Camtepe, Seyit and Nepal, Surya and Xu, Chang},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
