# Handwritten digits classificationusing MNIST datasets

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
  - Image size : 28 * 28 * 1
  
### Test Size:
  - Total number of images  10,000
  - Batch size : 32
  - Image size : 28 * 28 * 1

## DNN Model

For python code, please refer this notebook : [Click here](https://github.com/Noopuragr/EVA4/blob/master/S6/S6_Code_final_40epochs.ipynb)
  - Total number of parameters : 8,286
  
![Model](https://github.com/Noopuragr/EVA4/blob/master/S6/S6_model.PNG)

* After model training, display 25 misclassified images for L1 and L2 models.

## Misclassified image using L1 Regularization:
![L1](https://github.com/Noopuragr/EVA4/blob/master/S6/L1.PNG)

## Misclassified image using L2 Regularization:

![L2](https://github.com/Noopuragr/EVA4/blob/master/S6/L2.PNG)

## Loss Change function :
 This is loss vs epochs curve. It shows how the loss is decreasing as per epochs.
![Loss function](https://github.com/Noopuragr/EVA4/blob/master/S6/loss_function.PNG)
 
 ## Validation Accuracy change :
  This is loss vs epochs curve. It shows how the accuracy is increasing as per epochs.
 ![Accuracy](https://github.com/Noopuragr/EVA4/blob/master/S6/val_acc.PNG)
 
### Observation : 

- Without any loss functions the code ran for 40 epochs and the best value of loss obtained was ~0.07.

- L1 loss value seemed to be diminishing well. Though more iterations through different loss values can give us better results.

- Out of all the loss functions used we got L1 as the best one.

- L2 loss, the values seems to be spiking up and down. Values were taken as 5,0.5.0.05,0.005 and finally decided on 0.0003 value but still the peaks are observed.

- For both L1 and L2 loss implementation the values seem to steadyily moving across though our intuition says that the values might have to be lesser.

- In real world applications mostly L2 are used. 
