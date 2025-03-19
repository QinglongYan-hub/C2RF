#  C2RF

This is official Pytorch implementation of "**C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning**"(IJCV 2025)

## 1. Recommended Environment
 - [ ] torch  1.10.2+cu102
 - [ ] torchvision 0.8.2 
 - [ ] kornia 0.5.2

## 2. Framework
![The framework of the proposed C2RF for multi-modal image registration and fusion.](https://github.com/QinglongYan-hub/C2RF/blob/main/C2RF/Framework.png)
The framework of the proposed C2RF for multi-modal image registration and fusion.
## 3. To Test
### Registration and Fusion 
#### RoadScene dataset    
    python test.py --dataset=RoadScene 
#### PET-MRI dataset
    python test.py --dataset=PET-MRI

## 4. To Train
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
