# Model Inversion Attack on the Yale Faces Dataset

[Code](https://colab.research.google.com/drive/1PfzrrdE4TiiHtKjlOzTZql60zKZbVJeS?usp=sharing)

This experiment was mainly carried out to find out if a face recognition model that was already fully trained can reconstruct an input the model really strongly links with a chosen person through the use of its trained feature vector or weights.

Instead of beginning the attack with a real photograph, it all ends up starting with a random grey-scale noise picture. This initial picture is further modified through model gradients. Their objective is to make a model output of chosen category probability as large as possible, while at the same time generated image has to be relatively smooth.
We study two privacy threats:



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

## Face Classification Model: 
A small convolutional neural network (CNN) was trained before carrying out this experiment.
Our traget Model Architecture is :
```

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_3 (Conv2D)               │ (None, 64, 64, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 32, 32, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 32, 32, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_4 (MaxPooling2D)  │ (None, 16, 16, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)               │ (None, 16, 16, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)  │ (None, 8, 8, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_1 (Flatten)             │ (None, 8192)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 128)            │     1,048,704 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 15)             │         1,935 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 1,143,311 (4.36 MB)

 Trainable params: 1,143,311 (4.36 MB)

 Non-trainable params: 0 (0.00 B)
```
``` NOTE```:  In this experiment, we didn't use model structure and its parameter

Model was compiled with:
```
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```
Dataset spliting 
```
X_train: (132, 64, 64, 1)
X_test : (33, 64, 64, 1)

```
### Training Result

At the end of training:

Training accuracy: 1.0000
Validation accuracy: 0.8889

The model was then evaluated on the held out test set.

Test accuracy: 0.9090909

So the classifier reached approximately 90.91% test accuracy.



## 2. Model Inversion Attack : 
The purpose of the model inversion experiment is to investigate whether information associated with a target identity can be recovered from the trained classifier.
The attack starts with a randomly generated grayscale image and optimizes the input pixels, while keeping the trained CNN parameters fixed.
```

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

The attack does not use a real face as its starting input. Instead, a random image was generated with the shape:
```
random_image = tf.random.uniform(
    shape=(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    ),
    minval=0.0,
    maxval=1.0
)
```
(1, 64, 64, 1)

and pixel values between 0 and 1.

When this random image was sent to the classifier, the target-class confidence was only:

```
Random image target confidence: 0.0048025716
Random predicted subject: subject06

```
The model predicted: subject06
This gives a useful starting point: the random image does not initially look like the selected target to the classifier.

### Attack Parameters Used
Target class                       0
Target subject             subject01
Image size                   64 × 64
Maximum iterations              2000
Learning rate                   0.05
Confidence threshold            0.99
Total variation weight         0.001
```
model_inversion_attack(
        model=model,
        target_class=target_class,
        img_size=IMG_SIZE,
        max_iter=2000,
        learning_rate=0.05,
        threshold=0.99,
        tv_weight=0.001
    )
```
### Main Attack Result
The Attack start with

Iteration 0,
Target: subject01,
Confidence: 0.591375,
The attack then reported: Attack converged at iteration 10

```
Iteration    0 | Target: subject01 | Confidence: 0.591375

Attack converged at iteration 10

```

## Attack Result
I choose target_class = 0 and get the result after attack
```
Real subject: subject01
Target confidence: 0.99998546
Predicted subject: subject01
```
<img width="481" height="545" alt="image" src="https://github.com/user-attachments/assets/eb3d1f54-c4ba-4e7a-9efa-0640d3f7d408" />

## Experimental visualisation
<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/047f7a88-2d55-4bcd-85ff-1bdee84f3c8c" />
The graph reflects the boosting target confidence of the target class in the process of model inversion. From the initial randomly chosen picture, the confidence level is about 59%. The image undergoes modification by the gradient based optimizer and the change causes the confidence level of subject01 to increase quickly to around 100%, passing the 99% mark. After reaching a certain amount of iterations, the attack is stopped here at iteration 10 following the minimum-iteration condition in the implementation.

## Conclusion
The ultimately goal is that the attack generated random pixels and then gradually changed them until the network clearly preferred the chosen identity.

A major part of information that the network has about the target class is encoded in its learned representation so that an input can be optimized toward the class without having one of the training images of the class as part of the attack input. Even so, this particular set of results has no bearing in determining the accuracy of the original training picture.
The recovered photograph is a model input that has been optimally adjusted for the wanted model output. It is likely that such an input may display distinguishable features of the target class but not in the form of an exact reproduction of a particular training example.

    
