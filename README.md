# DMRA_RGBD-SOD
Code repository for our paper entilted "Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection" accepted at ICCV 2019 (poster).

# Overall
![avatar](https://github.com/jiwei0921/DMRA/blob/master/figure/overall.png)

## DUT-RGBD Dataset 
We create a large scale RGB-D dataset with 1200 paired images containing more complex scenarios, such as multiple or transparent objects, similar foreground and background, complex background, low-intensity environment. This challenging dataset can contribute to comprehensively evaluating saliency models.      

![avatar](https://github.com/jiwei0921/DMRA/blob/master/figure/dataset.png)
+ The **dataset link** can be found [here](https://pan.baidu.com/s/1FwUFmNBox_gMZ0CVjby2dg). And we split the dataset including 800 training set and 400 test set.   

## DMRA Code

### > Requirment
+ pytorch 0.3.0+
+ torchvision
+ PIL
+ numpy

### > Usage
#### 1. Clone the repo
```
git clone 
cd DMRA/
```
#### 2. Train/Test
+ test     
Download related dataset [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), and set the param '--phase' as "**test**" and '--param' as '**True**' in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.
```
python demo.py
```
+ train     
Our train-augment dataset [**link**](https://pan.baidu.com/s/18nVAiOkTKczB_ZpIzBHA0A)[fetch code **haxl**] / [train-ori dataset](https://pan.baidu.com/s/1B8PS4SXT7ISd-M6vAlrv_g), and set the param '--phase' as "**train**" and '--param' as '**True**'(loading checkpoint) or '**False**'(no loading checkpoint) in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.  
```
python demo.py
```

### > Train info and pre-trained models for DMRA
To better understand, we retrain our network and record some detailed training details as well as corresponding pre-trained models.

**Datasets** | - | NJUD(F-measure) | NJUD(MAE) | NLPR(F-measure) | NLPR(MAE)   
:-: | :-: | :-: | :-: | :-: | :-: |   
**Size** | 1985 | 1200 | 1000 | 929 | 797/1000 |    

+ Tips: **The results of the paper shall prevail.** Because of the randomness of the training process, the results fluctuated slightly.


### > Results  
| [DUT-RGBD](https://pan.baidu.com/s/1mS9EzoyY_ULXb3BCSd21eA)  |
| [NJUD](https://pan.baidu.com/s/1smz7KQbCPPClw58bDheH4w)  |
| [NLPR](https://pan.baidu.com/s/19qJkHtFQGV9oVtEFWY_ctg)  |
| [STEREO](https://pan.baidu.com/s/1L11R1c51mMPTrfpW6ykGjA)  |
| [LFSD](https://pan.baidu.com/s/1asgu1fGsHRk4CZcbz0NYxA)  |
| [RGBD135](https://pan.baidu.com/s/1jRYgoAijf_digGLQnsSbhA)  |
| [SSD](https://pan.baidu.com/s/1VY4I-4qpWS3wewz0MC8kqA)  |
+ Note:  For evaluation, all results are implemented on this ready-to-use [toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).
  
### > Related RGB-D Saliency Datasets
All common RGB-D Saliency Datasets we collected are shared in ready-to-use manner.       
+ The web link is [here](https://github.com/jiwei0921/RGBD-SOD-datasets).

### If you think this work is helpful, please cite
```
@InProceedings{Piao_2019_ICCV,       
   author = {Yongri {Piao} and Wei {Ji} and Jingjing {Li} and Miao {Zhang} and Huchuan {Lu}},   
   title = {Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection},     
   booktitle = "ICCV",     
   year = {2019}     
}  
```

### Contact Us
If you have any questions, please contact us ( jiwei521@mail.dlut.edu.cn ).
