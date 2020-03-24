# Session 9 - Data Augmentation and Grad Cam

The model reaches a maximum accuracy of **91.53%** in 20th epochs on CIFAR-10 dataset using **ResNet-18** model.

Gradient-weighted Class Activation Map (GradCAM) was implemented for each convolution block to generate model prediction heatmaps (Examples shown below).
Used resnet model.

Total params: 11,173,962.


**Implemented Albumentation** 

Rotate((-7.0, 7.0)),.

CutOut(),.
HorizontalFlip(),.

Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]).

Implemented GradCam.

![Gradcam](https://github.com/Noopuragr/EVA4/blob/master/S9/grad.PNG)

