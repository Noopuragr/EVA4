# Session 4 - Architectural Basics (MNIST Classification)

**Contents**

In this code the design of the convolution network is approached to achieve the target validation accuracy of minimum 99.4% with less than 20k parameters and 20 epochs. The below mentioned are the steps taken to reach the target.
1.	Network with less than 20k parameters achieved with having the number of kernels (3*3) as 16.
2.	Adding Batch Normalization and Dropout.
3.	Keeping the batch size to 128.

**Model Architecture**

**Details** :

1.	Total no. of parameters : 17,722
2.	Validation accuracy : 99.47 (16th epoch)
