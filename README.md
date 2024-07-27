# Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization
This repository contains code for our TMM paper ["Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization"](https://ieeexplore.ieee.org/abstract/document/10306334).

# Dependencies
We were using PyTorch 1.10.0 for all the experiments. You may want to install other versions of PyTorch according to the cuda version of your computer/server.
The code is run and tested on the Artemis HPC server with multiple GPUs. Running on a single GPU may need adjustments.

# Data and pre-trained models
We used the Microsoft COCO dataset to pre-train the mentioned styler transfer network and the CASIA-WebFace dataset to pre-train the face regconition networks. We used LFW, VGGFace2, AgeDB and CFP as the tested datasets. 

# Usage
Examples of training and evaluation scripts can be found in `train_stymask.py` and `test_style_attack.py`.

# Reference
If you find our paper/this repo useful for your research, please consider citing our work.
```
@article{gong2023stealthy,
  title={Stealthy Physical Masked Face Recognition Attack via Adversarial Style Optimization},
  author={Gong, Huihui and Dong, Minjing and Ma, Siqi and Camtepe, Seyit and Nepal, Surya and Xu, Chang},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
