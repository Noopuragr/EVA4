# Session 5 - Coding Drill Down

The goal of this assignment is to reach a test accuracy of 99.4% on the MNIST test dataset with a model having following configurations:

- Less than 10,000 parameters
- The desired accuracy should be achieved in 15 epochs
- The desired target should be achieved in a minimum of 5 steps.

## Step 1

[Step1 code](https://github.com/Noopuragr/EVA4/blob/master/S5/Step1/Step_1.ipynb)

###### Targets:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop

###### Results:
- Total params: 194,884
- Best Training Accuracy: 99.47
- Best Test Accuracy: 98.91

###### Analysis:
- Extremely Heavy Model for such a problem
- Model is over-fitting, but we are changing our model in the next step
