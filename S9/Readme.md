# Image classification using CIFAR dataset

## Aim :
* To perform image classification using CIFAR dataset under following conditions
 - Using Resnet-18 model
 - Above 90 % accuracy
 - Within 25 epochs 
* Perform albumentations
* Plot GradCAM for misclassified images
* To plot accuracy vs epochs curve

## Data description
Around 60K images of 10 different classes : ship, bird,plane, automobile, cat, deer, dog, frog, horse, truck

**Train data**
- Image size : 32 * 32 * 1
- Total images : 50000

**Test data**
- Image size : 32 * 32 * 1
- Total images : 10000 

**Implemented Albumentation** 

- Rotate((-7.0, 7.0))
- CutOut()
- HorizontalFlip(),.
- Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]).

**Images after performing cutout**
- ![](https://github.com/Noopuragr/EVA4/blob/master/S9/Cutout/cutout.png).
- ![](https://github.com/Noopuragr/EVA4/blob/master/S9/Cutout/cutout2.png).
- ![](https://github.com/Noopuragr/EVA4/blob/master/S9/Cutout/cutout3.png).
- ![](https://github.com/Noopuragr/EVA4/blob/master/S9/Cutout/cutout4.png).


## Model Structure

![](https://github.com/Noopuragr/EVA4/blob/master/S9/S9_model.PNG)


**Missclassified images**
![Misclassified_images](https://github.com/Noopuragr/EVA4/blob/master/S9/Cutout/misclassified.png)

**Implemented GradCam**
 -Implemented GradCam with the help of ['https://github.com/vickyliin/gradcam_plus_plus-pytorch/tree/master/gradcam'] 
 - Modified the code to show the activations of all the 4 layers.

![Gradcam](https://github.com/Noopuragr/EVA4/blob/master/S9/GradCam.PNG)

## Result
The model reaches a maximum accuracy of **91.53%** in 20th epochs on CIFAR-10 dataset using **ResNet-18** model.

**Accuracy plot**
![Accuracy plot](https://github.com/Noopuragr/EVA4/blob/master/S9/acc_plot.PNG)

