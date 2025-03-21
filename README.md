# LFTDet
Source Code for '**A Fourier-Transform-Based Framework With Asymptotic Attention for Mobile Thermal InfraRed Object Detection**' 

Accepted by **IEEE Sensors Journal**

![image](https://github.com/user-attachments/assets/9aaa1b38-a33d-462f-b4bf-2961b063973d)

This repo highly inherits the **mmdetection** framework.

# Abstract
Thermal InfraRed (TIR) technology has emerged as a significant tool in autonomous driving systems. Unlike natural images, TIR images are distinguished by their enriched thermal and illumination information while lacking chromatic contrast. Traditional object detection on natural images normally uses deep neural networks based on convolutional layers or attention modules. However, TIR-based object detection necessitates high computational efficiency to eliminate the extraction of redundant chromatic features. Furthermore, the robust space–frequency perception and expansive receptive field are critical due to the distinct brightness and contour features of TIR images. In this article, we propose a novel network, namely a lightweight Fourier-transform detector (LFTDet), meticulously designed to strike a balance between computational efficiency and accuracy in TIR object detection. Specifically, our innovative Fourier transform-efficient layer aggregation network (FT-ELAN) backbone takes advantage of Fourier transform (FT) in synergy with deep neural networks. In addition, we propose the detection neck called asymptotic attention-based feature pyramid network (AA-FPN) that integrates the SimA mechanism in the asymptotic structure to facilitate the FT-based operation. Extensive experiments conducted on FLIR and LLVIP datasets demonstrate that LFTDet surpasses all baselines while maintaining an extremely low computational cost. The code is available at https://github.com/zeyuwang-zju/LFTDet.

# Requirements
- torch=1.9.1 
- torchvision=0.9.1 
- cuda=11.1
- mmdet=2.28.2

Follow the implementations of **mmdetection** to train and test our model.

# Updates
We found a typo in TABLE I, where the n_ft and depth are confused. TABLE I should be:
![8aa465d1232d23a62a51d934f031380](https://github.com/user-attachments/assets/eb3abec1-515e-4a3b-9e61-8471b6773bd1)



## Citation
If you make use of our work, please cite our paper:
```
@article{wang2024fourier,
  title={A Fourier-Transform-Based Framework with Asymptotic Attention for Mobile Thermal InfraRed Object Detection},
  author={Wang, Zeyu and Shen, Haibin and Jiang, Wenyu and Huang, Kejie},
  journal={IEEE Sensors Journal},
  year={2024},
  publisher={IEEE}
}
```
