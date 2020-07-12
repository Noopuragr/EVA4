# Aim : 
1. To detect the objects in the given image using YoLov3 model.
2. To train YoloV3 model on custom dataset and predict the object from video. 


## Implementation

1. To detect the objects in the given image using YoLov3 model.

The link of implementation code : ![](https://github.com/Noopuragr/EVA4/blob/master/S13/Assigment1/Assigment%201.ipynb)

* Used openCV Darknet model.
* YOLO predicts detection on a 13x13 feature map. This may not be sufficient to detect smaller objects. To fix this, they simply add a pass-through layer that brings features from an earlier layer at 26x26 resolution. The pass-through layer concatenates the higher resolution features with the low-resolution features by stacking adjacent features into different channels.
* Every 10 batches, the network chooses a random new image dimension size (multiples of 32) from 320x320 to 608x608.
* The final model, called Darknet-19 has 19 convolution layers and 5 max-pooling layers. 1x1 convolutions are used to compress the feature representations between 3x3.
* The network is first trained on classification for 160 epochs.
* After classification training, the last convolution layer is removed, and three 3x3 convolution layers with 1024 filters each followed by the final 1x1 convolution layer are added. The network is again trained for 160 epochs.
* During training both, detection and classification datasets are mixed. When the network sees an image with detection label, full back-propagation is performed, else only the classification part is back-propagated.
