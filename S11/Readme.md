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
6. Plot for LR vs epochs ![](https://github.com/Noopuragr/EVA4/blob/master/S11/Images/LR_vs_Epochs.PNG)
