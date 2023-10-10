# Multimodal Dual-Swin Transformer with Spectral-Spatial Feature Extraction for Terrain Recognition
	
This project proposes a dual-branch SWIN transformer-based terrain recognition system with Multimodal Input. A User can input either an RGB or a Hyperspectral image into the application. The proposed model uses two SWIN Transformers, one for an RGB image, and the other for an HSI image. The RGB branch utilizes a spatial (texture) feature extraction method, while the proposed Hyperspectral branch adopts a joint spatial-spectral feature extraction technique and an adaptive spatial-spectral clustering algorithm for feature selection. 

The SWIN Transformer utilizes the attention mechanism to effectively capture short and long-range dependencies in the Image, providing state-of-the-art classification accuracy of 99% in the prototype trained on the RGB dataset. The prototype uses statistical variance-based texture feature analysis to estimate terrain roughness and slipperiness. The proposed HSI branch will not only provide spatial information like roughness and slipperiness but also the spectral signature for each pixel in the scene, providing a high-level environmental perception. 

This project's objective is to offer an effective and robust terrain recognition solution that can accept different types of images, depending on their availability. We aim to establish a compromise between computational complexity for terrain recognition and high-level implicit properties estimation by providing a dual framework, making this project an indispensable tool for remote sensing, environmental monitoring, and defense applications. 

This Repository contatins code built for this project, which was implemented using PyTorch Deep Learning framework.


## Prototype

### Data

This prototype was trained on an RGB remote sensing dataset having approximately 45 thousand images, with more than 10 thousand images for each terrain class (Grassy, Marshy, Rocky, Sandy). The Dataset can be downloaded from [here](https://www.kaggle.com/datasets/atharv1610/terrain-recognition)


### Training Procedure

#### Data Augmentation

The training data is augmented using techniques like jitter, crop, horizontal and vertical flip to increase diversity.

#### SWIN Transformer Model

The SWIN Transformer,which is pretrained on the ImageNet Dataset, is used as the base classification model. The SWIN Transformer serves as a versatile backbone for computer vision applications. This transformative architecture operates as a hierarchical Transformer, and its distinctive feature involves the computation of representations with the utilization of shifted windows. 

This windowing scheme enhances computational efficiency by restricting self-attention calculations to non-overlapping local windows. Simultaneously, it facilitates cross-window connections, yielding substantial gains in terms of both speed and memory usage. By adopting this model, we achieve a significant reduction in computational complexity, scaling from the expensive O($n^2$) of a traditional Vision Transformer to a much more manageable O($mn$), for reasonably small window size ($m$).

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/SWIN%20Attention.gif">
</p>


#### Model Training

The model is compiled with the AdamW optimizer and label smoothing cross-entropy loss. It is then trained for 10 epochs.

#### Training History

| Epoch | Loss     | Accuracy | Validation Loss | Validation Accuracy |
|-------|----------|----------|-----------------|---------------------|
| 0     | 0.4909   | 0.9455   | 0.4088          | 0.9854              |
| 1     | 0.4096   | 0.9871   | 0.3912          | 0.9923              |
| 2     | 0.3962   | 0.9921   | 0.3815          | 0.9941              |
| 3     | 0.3904   | 0.9941   | 0.3800          | 0.9950              |
| 4     | 0.3855   | 0.9960   | 0.3778          | 0.9953              |
| 5     | 0.3835   | 0.9960   | 0.3746          | 0.9965              |
| 6     | 0.3813   | 0.9969   | 0.3756          | 0.9963              |
| 7     | 0.3809   | 0.9970   | 0.3742          | 0.9967              |
| 8     | 0.3790   | 0.9978   | 0.3711          | 0.9976              |
| 9     | 0.3769   | 0.9982   | 0.3713          | 0.9972              |


### Training Results

#### Accuracy

The model achieved an exceptional test accuracy of 99%, showcasing its robust performance in classifying terrain types.

#### Confusion Matrix

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/prototype%20confusion%20matrix.png">
</p>






## Applications


#### 1. Defence

   - **Tactical Planning:** 
   
   - **Vehicle and Equipment Deployment.:** 
   
   
#### 2. Environmental Monitoring

   - **Conservation Efforts:** 
   
   - **Disaster Response:** 
   

These applications demonstrate the broad utility of the proposed application across different domains.

	
Based on the problem statement, two primary tasks are identified: Terrain Recognition (Multiclass scene classification) and the estimation of implicit properties (pixel/patchwise property inference). While traditional Deep Learning methods like CNN can effectively handle Multiclass Terrain classification, the estimation of implicit properties demands a more sophisticated approach, making it a primary focus. In Remote sensing, terrain roughness is traditionally assessed using Lidar (Light Detection and Ranging) to create Digital Elevation Maps (DEM). However, Lidar has limitations, including environmental distortion and a relatively short range, rendering it unsuitable for satellite-based defense applications. Hyperspectral Imaging offers a solution to this problem by providing rich spatial and spectral resolution, making it resilient to environmental distortion. A Hyperspectral image captures reflectance information across hundreds of narrow, contiguous wavelength bands, making it ideal for estimating implicit properties based on the spectral signatures of each pixel. This project proposes a versatile multimodal application designed to recognize various terrains. It accepts both RGB and Hyperspectral images as input and aims to recognize the terrain and estimate its implicit properties. A dual branch shifted window image transformer (Swin) Architecture is proposed. For RGB input, spatial features like texture are extracted for terrain recognition and implicit property estimation. For Hyperspectral input, a joint spectral-spatial feature extraction method is proposed, along with an adaptive spatial-spectral clustering algorithm for feature selection, given the high dimensionality of HSI images. For RGB input, the desired output includes the predicted terrain class, coupled with estimated roughness and slipperiness achieved through spatial analysis. In the case of HSI input, the desired output is the predicted terrain class and implicit properties information such as roughness, slipperiness, and the spectral signature for each pixel or patch using spatial and spectral analysis. The project prototype, which is trained on an RGB Dataset, achieves a test accuracy of 99% for the terrain recognition task, and terrain roughness and slipperiness information is calculated using a statistical variance-based patch texture analysis and is visualized as an overlay over the original image. The prototype can be viewed here: https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023

