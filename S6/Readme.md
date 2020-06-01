# Batch Normalization & Regularization

## Aim :

* To perform handwritten digits classification using MNIST dataset with 99.4 % accuracy in less then 40 epochs.
* To explore the performance of DNN by applying different regularization techniques. 

  - Without L1 and L2 regularization.
  - With L1 regularization.
  - With L2 regularization.
  - With L1 and L2 regularization.
  
## Data description
### Train data :
  - Total number of images : 60,000
  - Batch size : 32
  
### Test Size:
  - Total number of images  10,000
  - Batch size : 32

## DNN Model
  - Total number of parameters : 8,286
  -         Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 28, 28]              90
              ReLU-2           [-1, 10, 28, 28]               0
       BatchNorm2d-3           [-1, 10, 28, 28]              20
           Dropout-4           [-1, 10, 28, 28]               0
            Conv2d-5           [-1, 16, 28, 28]           1,440
              ReLU-6           [-1, 16, 28, 28]               0
       BatchNorm2d-7           [-1, 16, 28, 28]              32
           Dropout-8           [-1, 16, 28, 28]               0
            Conv2d-9           [-1, 10, 28, 28]             160
        MaxPool2d-10           [-1, 10, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           1,440
             ReLU-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
           Conv2d-14           [-1, 16, 14, 14]           2,304
             ReLU-15           [-1, 16, 14, 14]               0
      BatchNorm2d-16           [-1, 16, 14, 14]              32
          Dropout-17           [-1, 16, 14, 14]               0
           Conv2d-18           [-1, 16, 14, 14]             256
        MaxPool2d-19             [-1, 16, 7, 7]               0
           Conv2d-20             [-1, 10, 7, 7]           1,440
             ReLU-21             [-1, 10, 7, 7]               0
      BatchNorm2d-22             [-1, 10, 7, 7]              20
          Dropout-23             [-1, 10, 7, 7]               0
           Conv2d-24             [-1, 10, 7, 7]             900
             ReLU-25             [-1, 10, 7, 7]               0
      BatchNorm2d-26             [-1, 10, 7, 7]              20
          Dropout-27             [-1, 10, 7, 7]               0
        AvgPool2d-28             [-1, 10, 1, 1]               0
           Conv2d-29             [-1, 10, 1, 1]             100

* After model training, display 25 misclassified images for L1 and L2 models.

**Misclassified image using L1 Regularization:**
![L1](https://github.com/Noopuragr/EVA4/blob/master/S6/L1.PNG)

**Misclassified image using L2 Regularization:**

![L2](https://github.com/Noopuragr/EVA4/blob/master/S6/L2.PNG)

**Loss Change function :**

![Loss function](https://github.com/Noopuragr/EVA4/blob/master/S6/loss_function.PNG)
 
 **Validation Accuracy change :**
 
 ![Accuracy](https://github.com/Noopuragr/EVA4/blob/master/S6/val_acc.PNG)
 
### Observation : 




- Without any loss functions the code ran for 40 epochs and the best value of loss obtained was ~0.07.

- L1 loss value seemed to be diminishing well. Though more iterations through different loss values can give us better results.

- Out of all the loss functions used we got L1 as the best one.

- L2 loss, the values seems to be spiking up and down. Values were taken as 5,0.5.0.05,0.005 and finally decided on 0.0003 value but still the peaks are observed.

- For both L1 and L2 loss implementation the values seem to steadyily moving across though our intuition says that the values might have to be lesser.

- In real world applications mostly L2 are used. 
