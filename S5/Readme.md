# Session 5 - Coding Drill Down

The goal of this assignment is to reach a test accuracy of 99.4% on the MNIST test dataset with a model having following configurations:

- Less than 10,000 parameters
- The desired accuracy should be achieved in 15 epochs
- The desired target should be achieved in a minimum of 5 steps.

## Step 1

[Step1 code](https://github.com/Noopuragr/EVA4/blob/master/S5/Step1/Step_1.ipynb)

**Targets:**

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop

**Results:**
- Total params: 194,884
- Best Training Accuracy: 99.47
- Best Test Accuracy: 98.91

**Analysis:**
- Extremely Heavy Model for such a problem
- Model is over-fitting, but we are changing our model in the next step

## Step 2

[Step2 code](https://github.com/Noopuragr/EVA4/blob/master/S5/Step%202/Step_2.ipynb)

**Targets:**

- Get the basic skeleton right.
- To get less than 10 K parameters.

**Changes done**
- Changed the size of kernels to get the required parameters.

**Results:**
- Total params: 9,448
- Best Train Accuracy: 99.16
- Best Test Accuracy: 98.63

**Analysis:**
-  Getting parameters in range but we need to improve accuracy.

## Step 3

[Step3 code](https://github.com/Noopuragr/EVA4/blob/master/S5/Step_3/Step_3.ipynb)

**Targets:**

- To reduce overfitting.
- Reduce gap between train and test accuracy.

**Changes done**
- Added Batch normalization and dropout.

**Results:**
- Total params: 9,588
- Train Accuracy : 99.24
- Test Accuracy : 99.24

**Analysis:**
-  Good Model.
- No overfitting, model is capable if pushed further.
- Still need to improve accuracy

## Step 4

[Step4 code](https://github.com/Noopuragr/EVA4/blob/master/S5/Step_4/Step_4.ipynb)

**Targets:**

- To improve the accuracy.

**Changes done**
- Add GAP layer

**Results:**
- Parameters: 6k.
- Best Train Accuracy: 98.82
- Best Test Accuracy: 99.11

**Analysis:**
- After adding GAP, parameters are reduced.
- We need to increase the capacity to improve accuracy.

## Step 5

[Step5 code](https://github.com/Noopuragr/EVA4/tree/master/S5/Step_5)

**Targets:**

- To increase the capacity and accuracy.

**Changes done**
- Increased kernel size
- Added Rotation
- Added Scheduler

**Results:**
- Total params: 8,992
- Best Train Accuracy : 99.05
- Best Test Accuracy: 99.41

**Analysis:**
- After 8th Epoch accuracy is contantly above 99.3
- At 12th epoch it reached 99.4
- We need to increase the capacity to improve accuracy.
