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

Official Implementation of the swin transformer can be found [here](https://github.com/microsoft/Swin-Transformer)

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/SWIN%20Attention.gif">
</p>


#### Model Training

The model is compiled with the AdamW optimizer and label smoothing cross-entropy loss. It is then trained for 10 epochs.

#### Training History

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/model%20training%20history.png">
</p>

### Training Results

#### Accuracy

The model achieved an exceptional test accuracy of 99%, showcasing its robust performance in classifying terrain types.

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/test%20accuracy.png">
</p>


#### Confusion Matrix

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/prototype%20confusion%20matrix.png">
</p>

### Roughness and Sliperiness Estimation

Implicit properties like roughness and sliperiness are estimated using a statistical approach (variance-based patch texture analysis). 

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/texture%201.png">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/texture%202.png">
</p>

The image is divided into patches, and variance of each patch is calculated. Then, based on the roughness factor value, terrain roughness and smoothness is estimated. 

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/implicit%20properties%20via%20texture%20variance.png">
</p>

## Applications


#### 1. Defence

   - **Tactical Planning** 
   
   - **Vehicle and Equipment Deployment** 
   
   
#### 2. Environmental Monitoring

   - **Conservation Efforts** 
   
   - **Disaster Response**
     
## Proposed Project  

This is the architecture of the proposed dual branch swin transformer.

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/architecture.png">
</p>

This is the block diagram of the Hyperspectral branch.

<p align="center">
	<img src="https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/HSI%20Branch.png">
</p>


