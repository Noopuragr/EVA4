
# Aim : 
1. To detect the objects in the given image using YoLov3 model.
2. To train YoloV3 model on custom dataset and predict the object from video. 


# Implementation

## 1. To detect the objects in the given image using YoLov3 model.

* Used pre-trainned openCV Darknet model.
* YOLO predicts detection on a 13x13 feature map. This may not be sufficient to detect smaller objects. To fix this, they simply add a pass-through layer that brings features from an earlier layer at 26x26 resolution. The pass-through layer concatenates the higher resolution features with the low-resolution features by stacking adjacent features into different channels.
* Every 10 batches, the network chooses a random new image dimension size (multiples of 32) from 320x320 to 608x608.
* The final model, called Darknet-19 has 19 convolution layers and 5 max-pooling layers. 1x1 convolutions are used to compress the feature representations between 3x3.

The link of implementation code : !(https://github.com/Noopuragr/EVA4/blob/master/S13/Assigment1/Assigment%201.ipynb)

The final output is : ![](https://github.com/Noopuragr/EVA4/blob/master/S13/Assigment1/result_opencv.PNG)

![](https://github.com/Noopuragr/EVA4/blob/master/S13/Assigment1/test_img.png)


## 2. To train YoloV3 model on custom dataset and predict the object from video. 

* Collected 500 images of "Woody" , a famous character of cartoon series "Toy Story".
* Annotated all images using this [code](https://github.com/miki998/YoloV3_Annotation_Tool).
* Refered [this](https://github.com/theschoolofai/YoloV3) repo for implementation.
* Ran for 300 epochs. And then tried to predict the character in video.

### Trained Samples

![](https://github.com/Noopuragr/EVA4/blob/master/S13/Assigment2/trained_samples.PNG)

### Final output

Please refer this video for the file output  : (https://www.youtube.com/watch?v=y-qddG41SAY)

