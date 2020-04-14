# Super Convergence

In this session we implemented
1. New Resnet Architecture
2. One Cycle policy such that:
   * Total Epochs = 24
   * Max at Epoch = 5
   * LRMIN = FIND
   * LRMAX = FIND
   * NO Annihilation
3. RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Accuracy > 90 %
6. Trained images

![](https://github.com/Noopuragr/EVA4/blob/master/S11/Images/train_imgs.PNG)
7. Plot for LR vs Accuracy

![](https://github.com/Noopuragr/EVA4/blob/master/S11/Images/LR_vs_Acc.PNG)
8. Plot for LR vs epochs

![](https://github.com/Noopuragr/EVA4/blob/master/S11/Images/LR_vs_Epochs.PNG)
9. Plot for train and validation acc

![](https://github.com/Noopuragr/EVA4/blob/master/S11/Images/acc_plot.PNG)
