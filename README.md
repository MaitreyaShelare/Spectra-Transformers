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
![WhatsApp Image 2023-09-12 at 15 43 47](https://github.com/PiPlusTheta/EarthFinesse/assets/68808227/39b98fd1-ded6-4950-a06a-a76966865250)


#### Training History

| Epoch | Loss     | Accuracy | Validation Loss | Validation Accuracy |
|-------|----------|----------|-----------------|---------------------|
| 0     | 0.4909   | 0.9455   | 0.4088          | 0.9854              |
| 1     | 0.151308 | 0.945084 | 0.208954        | 0.924763            |
| 2     | 0.121970 | 0.956086 | 0.170471        | 0.941647            |
| 3     | 0.101868 | 0.962776 | 0.154959        | 0.947571            |
| 4     | 0.090680 | 0.967120 | 0.118927        | 0.961345            |
| 5     | 0.080031 | 0.970640 | 0.128688        | 0.959271            |
| 6     | 0.073431 | 0.974317 | 0.131562        | 0.957198            |
| 7     | 0.071057 | 0.974508 | 0.123268        | 0.961197            |
| 8     | 0.064471 | 0.977139 | 0.129367        | 0.958235            |
| 9     | 0.059202 | 0.978661 | 0.114494        | 0.966380            |



## Applications


#### 1. Defence

   - **Tactical Planning:** 
   
   - **Vehicle and Equipment Deployment.:** 
   
   
#### 2. Environmental Monitoring

   - **Conservation Efforts:** 
   
   - **Disaster Response:** 
   

These applications demonstrate the broad utility of the proposed application across different domains.


