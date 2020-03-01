
URL for Misclassified image for L1:  https://github.com/Noopuragr/EVA4/blob/master/S6/L1.PNG

URL for Misclassified image for L2:  https://github.com/Noopuragr/EVA4/blob/master/S6/L2.PNG

URL for Loss Change function : https://github.com/Noopuragr/EVA4/blob/master/S6/loss_function.PNG
 
 URL for Validation Accuracy change :   https://github.com/Noopuragr/EVA4/blob/master/S6/val_acc.PNG
 
 Observation : 
 
 We have used the code in 4 versions: 
 
1) No loss
2) L1 loss value
3) L2 Loss value
4) L1 + L2 loss value

Observations
• Without any loss functions the code ran for 40 epochs and the best value of loss obtained was ~0.07
• L1 loss value seemed to be diminishing well. Though more iterations through different loss values can give us better results
• Out of all the loss functions used we got L1 as the best one.
• L2 loss, the values seems to be spiking up and down. Values were taken as 5,0.5.0.05,0.005 and finally decided on 0.00005 value but still the peaks are observed.
• For both L1 and L2 loss implementation the values seem to steadyily moving across though our intuition says that the values might have to be lesser. 
• In real world applications mostly L2 are used. 
