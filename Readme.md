# Cat vs. Dog Classification with Data Augmentation

Group 7

1. Soham Gaonkar (23110314)
2. Chaitanya Sharma (23110072)

colab link : [Colab Link](https://colab.research.google.com/drive/1r2WO6V1zD3-b7O4WOQ1nABMP1FJhCU_M#scrollTo=6OA4FKjd1rYu)
Incase the github doesn't render the notebook properly, please use the above link to view the notebook in colab.

This repository demonstrates how data augmentation can improve the performance of a binary image classifier (Cats vs. Dogs) using the ResNet-50 model.

### Resnet50 Model
![alt text](Images/resnet50.png)

---

## Data Details

- **Class Mapping:**  
  - Cats: 0  
  - Dogs: 1

- **Dataset Sizes:**  
  - Original Training Data: 112 images  
  - Augmented Training Data: 336 images  
  - Test Data: 28 images

---

## Augmentations Applied

- **Blur:** Applies a blur with a radius of 2  
- **Rotate:** Rotates images by a random degree between 0 and 20  
- **Brightness:** Adjusts brightness with a factor between 0.5 and 1.5  
- **Saturation:** Alters saturation  
- **Scale:** Scales images by a factor between 0.5 and 1.5  
- **Opacity:** Changes opacity with a level between 0.6 and 0.9  
- **Shuffle Pixels:** Randomly shuffles pixels with a factor between 0.05 and 0.2  
- **Padding:** Pads images to make them square  
- **Random Noise:** Adds random noise  
- **Color Jitter:** Applies color jitter with random brightness, contrast, and saturation adjustments

#### Example :

Original Image:

Augment 1: Shuffle Pixels -> Padding -> Opacity -> Done

Augment 2: Brightness -> Opacity -> Scale -> Done

![Example Aug](Images/aug.png)
---

## Plots

### Task 1: Data Augmentation Statistics
![Stats1](Images/bar1.png)
![Stats2](Images/bar2.png)


### Task 2: Model Training
- Loads ResNet-50 pretrained model  
- Modifies classifier for binary classification (Cats vs. Dogs)  
- Initializes weights using Kaiming Normal  
- Saves initialized weights for later use  
- Moves model to GPU/CPU for computation  

```python
# Load ResNet-50 model
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)

# Modify classifier for binary classification (Cats vs. Dogs)
in_features = model.classifier[-1].in_features  # Access last layer's input features
model.classifier[-1] = nn.Linear(in_features, 2)  # Change to binary classification (2 classes)

# Function to initialize weights
def initialize_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# Apply initialization
model.apply(initialize_weights)

# Save initialized weights (before training)
torch.save(model.state_dict(), "initial_weights.pth")

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model initialized and classifier modified successfully.")
```

- **Training Loss Curves:**  
  ![Training Loss](Images/TrainLoss.png)

## Evaluation Metrics

### Without Augmentation
- **Accuracy:** 0.5000  
- **Precision:** 0.5000  
- **Recall:** 1.0000  
- **F1 Score:** 0.6667

### With Augmentation
- **Accuracy:** 0.5357  
- **Precision:** 0.5185  
- **Recall:** 1.0000  
- **F1 Score:** 0.6829

---

## Explanation

The low accuracy is likely due to the limited amount of data (112 original training images and 28 test images) and inconsistent image quality.

Although augmentation increased the training size, some augmentations introduced unrealistic variations (e.g., random sizes and images with multiple cats) that may confuse the model. 

Data augmentation increases the diversity of training examples, helping the model generalize better to unseen data ,hence its performance is better than without augmentation,but not significantly.

The perfect recall indicates nearly all dogs (class 1) are detected, but the low precision suggests many cats (class 0) are misclassified as dogs. Improving data quality and using more targeted augmentation strategies may enhance performance.

---
