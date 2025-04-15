#  C2RF
### [Paper](https://link.springer.com/article/10.1007/s11263-025-02427-1) |  [Code](https://github.com/QinglongYan-hub/C2RF) 

This is official Pytorch implementation of "**C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning**"

## 1. Recommended Environment
 - [ ] torch  1.10.2+cu102
 - [ ] torchvision 0.8.2 
 - [ ] kornia 0.5.2

## 2. Framework
The framework of the proposed C2RF for multi-modal image registration and fusion.
![The framework of the proposed C2RF for multi-modal image registration and fusion.](https://github.com/QinglongYan-hub/C2RF/blob/main/C2RF/Framework.png)

## 3. Pretrained Weights
Please download the pretrained weights at the link below, and then place them into the folder ./checkpoint/
- The pretrained weights for the Roadscene dataset is at [Google Drive](https://drive.google.com/drive/folders/1wOSVg9CsqZBJkHWYMGD1kCER9tThSxYk?usp=sharing).

- The pretrained weights for the PET-MRI dataset is at [Google Drive](https://drive.google.com/drive/folders/1M99NDvcnk71iZUVC6BlYyRvAKIZlUIK6?usp=sharing).

## 4. To Test
### Registration and Fusion 
#### RoadScene dataset    
    python test.py --dataset=RoadScene 
#### PET-MRI dataset
    python test.py --dataset=PET-MRI

## 5. To Train
### Training the fusion model 
#### RoadScene dataset
    python train_Fu.py --dataset=RoadScene
#### PET-MRI dataset
    python train_Fu.py --dataset=PET-MRI

### Training the registration model 
#### RoadScene dataset
    python train_Reg.py --dataset=RoadScene
#### PET-MRI dataset
    python train_Reg.py --dataset=PET-MRI
