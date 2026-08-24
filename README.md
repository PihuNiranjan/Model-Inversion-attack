# Model Inversion Attack and Membership Inference Attack on the Yale Faces Dataset

[Code](https://colab.research.google.com/drive/1PfzrrdE4TiiHtKjlOzTZql60zKZbVJeS?usp=sharing)

This project studies privacy leakage in a face-recognition neural
network trained on the **Yale Faces dataset**. The dataset contains 15
subjects, with multiple face images for each subject under different
expressions, lighting conditions, and accessories such as glasses.

We study two privacy threats:

1. **Model Inversion Attack (MIA-Inversion)** : trying to get backan image from a trained model which is very similar to some target identity face.

2. **Membership Inference Attack (MIA-Membership)** : guessing that face image is the part of model's training data set.

## 1. Dataset

The dataset I used for this experiment :

``` python
DATASET_DIR = "/content/Face-Recognition/yalefaces/yalefaces"
```
### Dataset description 
Total images: 165
Total labels: 165


Images per subject:

subject01 : 11 images
subject02 : 11 images
subject03 : 11 images
subject04 : 11 images
subject05 : 11 images
subject06 : 11 images
subject07 : 11 images
subject08 : 11 images
subject09 : 11 images
subject10 : 11 images
subject11 : 11 images
subject12 : 11 images
subject13 : 11 images
subject14 : 11 images
subject15 : 11 images

<img width="1189" height="780" alt="image" src="https://github.com/user-attachments/assets/424e5239-bbf0-46e3-b50f-e84e96a347a3" />



## 2. Model Inversion Attack : 
The purpose of the model inversion experiment is to investigate whether information associated with a target identity can be recovered from the trained classifier.
The attack starts with a randomly generated grayscale image and optimizes the **input pixels**, while keeping the trained CNN parameters fixed.
``` text
Random grayscale image
        |
    Trained CNN
        |
Target subject confidence
        |
Calculate gradient with respect to input
        |
Update input pixels
        |
     Repeat
```

For example, if the target is `subject01`, the optimization attempts to
increase:

``` text
P(subject01 | generated image)
```

I choose target_class = 0 and get the resutl after attack
```
Real subject: subject01
Target confidence: 0.99998546
Predicted subject: subject01
```
<img width="481" height="545" alt="image" src="https://github.com/user-attachments/assets/eb3d1f54-c4ba-4e7a-9efa-0640d3f7d408" />

    
