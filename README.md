# Multimodal Dual-Swin Transformer with Spectral-Spatial Feature Extraction for Terrain Recognition


We propose a dual branch swin transformer architecture for terrain recognition and implicit properties estimation. capable of accepting multi-modal input (RGB and Hyperspectral) which would be capable of accepting RGB and Hyperspectral Images as input. EarthFinesse is a high-accuracy military terrain classifier powered by deep learning. It classifies terrain types such as Grassy, Marshy, Rocky, and Sandy with an accuracy of over 97.87%, setting a new benchmark in this domain. The model uses the MobileNetV2 architecture, optimized for efficient and accurate terrain classification.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Training Procedure](#training-procedure)
- [Training Results](#training-results)
- [Applications](#applications)




## Model Training

### Dataset

The model was trained on a dataset consisting of 45.1k images, with more than 10k images for each terrain class (Grassy, Marshy, Rocky, Sandy).

![WhatsApp Image 2023-09-13 at 13 18 11](https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/65ab6221-7657-4dca-99e2-87ed4eb9036f)

### Training Procedure

#### Data Augmentation

The training data is augmented using techniques like shear, zoom, and horizontal flip to increase diversity.

#### MobileNetV2 Base Model

https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_10.37.14_PM.png

The MobileNetV2 architecture, pre-trained on ImageNet, is used as the base model for feature extraction. All base model layers are frozen to retain pre-trained knowledge.

#### Custom Classification Head

A custom classification head is added to the base model. It includes a global average pooling layer, a dense layer with 1024 units and ReLU activation, and a final dense layer with softmax activation for the number of classes (4 in this case).

#### Compilation and Training

The model is compiled with the Adam optimizer and categorical cross-entropy loss. It is then trained for 10 epochs.

Here's how the model was trained:

```python
# Data augmentation and generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# ... (similar setup for test and validation generators)

# MobileNetV2 base model
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# ... (freeze base_model layers and add custom classification head)

# Compilation and training
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

### Training Results

EarthFinesse achieved remarkable training results, setting a new benchmark in terrain classification:

#### Final Accuracy

The model achieved a stunning final accuracy of over 97.87%, showcasing its robust performance in classifying terrain types. This high accuracy can significantly enhance the effectiveness of military operations.

#### Confusion Matrix
![WhatsApp Image 2023-09-12 at 15 43 47]([assets/prototype confusion matrix.png](https://github.com/MaitreyaShelare/Spectra-Transformers-SIH-2023/blob/main/assets/prototype%20confusion%20matrix.png))


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



## Applications


#### 1. Defence

   - **Tactical Planning:** 
   
   - **Vehicle and Equipment Deployment.:** 
   
   
#### 2. Environmental Monitoring

   - **Conservation Efforts:** 
   
   - **Disaster Response:** 
   

These applications demonstrate the broad utility of the proposed application across different domains.


