# DMRA_RGBD-SOD
Code repository for our paper entilted "Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection" accepted at ICCV 2019 (poster).

# Overall
![avatar](https://github.com/jiwei0921/DMRA/blob/master/figure/overall.png)

## The proposed Dataset 
+ Dataset: DUTLF
1. This dataset consists of DUTLF-MV, DUTLF-FS, DUTLF-Depth.
2. The dataset will be expanded to 3000 about real scenes.
3. We are working on it and will make it publicly available soon. 
+ Dataset: DUTLF-Depth
1. The dataset is part of DUTLF dataset captured by Lytro camera, and we selected a more accurate 1200 depth map pairs for more accurate RGB-D saliency detection.     
2. We create a large scale RGB-D dataset(DUTLF-Depth) with 1200 paired images containing more complex scenarios, such as multiple or transparent objects, similar foreground and background, complex background, low-intensity environment. This challenging dataset can contribute to comprehensively evaluating saliency models.    

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
git clone https://github.com/jiwei0921/DMRA.git
cd DMRA/
```
#### 2. Train/Test
+ test     
Download related dataset [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), and set the param '--phase' as "**test**" and '--param' as '**True**' in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.
```
python demo.py
```
+ train     
Our train-augment dataset [**link**](https://pan.baidu.com/s/18nVAiOkTKczB_ZpIzBHA0A) [ fetch code **haxl** ] / [train-ori dataset](https://pan.baidu.com/s/1B8PS4SXT7ISd-M6vAlrv_g), and set the param '--phase' as "**train**" and '--param' as '**True**'(loading checkpoint) or '**False**'(no loading checkpoint) in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.  
```
python demo.py
```

### > Training info and pre-trained models for DMRA
To better understand, we retrain our network and record some detailed training details as well as corresponding pre-trained models.

**Iterations** | **Loss** | NJUD(F-measure) | NJUD(MAE) | NLPR(F-measure) | NLPR(MAE) | download link     
:-: | :-: | :-: | :-: | :-: | :-: | :-: |   
100W | 958 | 0.882 | 0.048 | 0.867 | 0.031 | [link](https://pan.baidu.com/s/1Hb0CDDH7vG6F9yxl6wTymQ)   
70W | 2413 | 0.876 | 0.050 | 0.854 | 0.033 | [link](https://pan.baidu.com/s/19SvkoKrkLPHFJUa_9z4ulg)  
40W | 3194 | 0.861 | 0.056 | 0.823 | 0.037 | [link](https://pan.baidu.com/s/1_1ihh0TIm9pwQ4nyNSXKDQ)   
16W | 8260 | 0.805 | 0.081 | 0.725 | 0.056 | [link](https://pan.baidu.com/s/1BzCOBV5HKNLAJcON0ImqyQ)  
2W | 33494 | 0.009 | 0.470 | 0.030 | 0.452 | [link](https://pan.baidu.com/s/1QUJsr3oPOCUJsJu8nCHbHQ)  
0W | 45394 | - | - | - | - | -  

+ Tips: **The results of the paper shall prevail.** Because of the randomness of the training process, the results fluctuated slightly.


### > Results  
| [DUTLF-Depth](https://pan.baidu.com/s/1mS9EzoyY_ULXb3BCSd21eA)  |
| [NJUD](https://pan.baidu.com/s/1smz7KQbCPPClw58bDheH4w)  |
| [NLPR](https://pan.baidu.com/s/19qJkHtFQGV9oVtEFWY_ctg)  |
| [STEREO](https://pan.baidu.com/s/1L11R1c51mMPTrfpW6ykGjA)  |
| [LFSD](https://pan.baidu.com/s/1asgu1fGsHRk4CZcbz0NYxA)  |
| [RGBD135](https://pan.baidu.com/s/1jRYgoAijf_digGLQnsSbhA)  |
| [SSD](https://pan.baidu.com/s/1VY4I-4qpWS3wewz0MC8kqA)  |
+ Note:  For evaluation, all results are implemented on this ready-to-use [toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).
+ SIP results: This is [test results](https://pan.baidu.com/s/1R126FXbZBE7Lj-B7nNae_g) on SIP dataset, and fetch code is 'fi5h'.
  
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


## Related SOTA RGB-D methods' results on our dataset

Meanwhile, we also provide other state-of-the-art RGB-D methods' results on our proposed dataset, and you can directly download their results ([All results](https://pan.baidu.com/s/1spuFNQl7FJiDPFSOS55lnw),2gs2).     

  
**No.** | **Pub.** | **Name** | **Title** | **Download**    
:-: | :-: | :-: | :- | :-: | 
14 | **ICCV2019** | **DMRA** | Depth-induced multi-scale recurrent attention network for saliency detection | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz
13 | **CVPR2019** | **CPFP** | Depth-induced multi-scale recurrent attention network for saliency detection | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
12 | **TIP2019** | **TANet** | Three-stream attention-aware network for RGB-D salient object detection | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
11 | **PR2019** | **MMCI** | Multi-modal fusion network with multiscale multi-path and cross-modal interactions for RGB-D salient object detection | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
10 | **ICME2019** | **PDNet** | Pdnet: Prior-model guided depth-enhanced network for salient object detection | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
09 | **CVPR2018** | **PCA** | Progressively Complementarity-Aware Fusion Network for RGB-D Salient Object Detection | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
08 | **ICCVW2017** | **CDCP** | An innovative salient object detection using center-dark channel prior | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
07 | **TCyb2017** | **CTMF** | CNNs-based RGB-D saliency detection via cross-view transfer and multiview fusion | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
06 | **TIP2017** | **DF** | RGBD salient object detection via deep fusion | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
05 | **CAIP2017** | **MB** | A Multilayer Backpropagation Saliency Detection Algorithm Based on Depth Mining | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
04 | **SPL2016** | **DCMC** | Saliency detection for stereoscopic images based on depth confidence analysis and multiple cues fusion | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
03 | **ECCV2014** | **LHM-NLPR** | Rgbd salient object detection: a benchmark and algorithms | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
02 | **ICIP2014** | **ACSD** | Depth saliency based on anisotropic center-surround difference | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 
01 | **ICIMCS2014** | **DES** | Depth enhanced saliency detection method | [results](https://pan.baidu.com/s/1Wg1rrO4bNV9kBcmirW-HXA), g7rz 

+ Thanks for related authors to provide the code or results, particularly, [Deng-ping Fan](http://dpfan.net), [Hao Chen](https://github.com/haochen593), [Chun-biao Zhu](https://github.com/ChunbiaoZhu), etc. 

### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca or weiji.dlut@gmail.com ).
