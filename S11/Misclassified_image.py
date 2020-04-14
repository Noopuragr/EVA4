import Utils
from GradCAM import GradCAM
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch
from Utils import visualize_cam
import torchvision
import torchvision.transforms as transforms

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow2(img,pred,target ):
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title("Predicted: %s\nActual: %s" % (classes[pred],classes[target]))



def gradcam_misclassified(mis_class,model,device):
  plt.figure(figsize=(75, 75))
  num_of_images = len(mis_class)
  for index in range(1, num_of_images + 1):
    images =[]
    b = torch.from_numpy(mis_class[index-1]['img'])
    trans = transforms.ToPILImage()
    pil_img=trans(torchvision.utils.make_grid(b))
    t_img,n_t_img=Utils.change(pil_img,device)
    g= GradCAM(model,model.layer4)
    mask,_=g(n_t_img)
    heatmap,res = visualize_cam(mask,t_img)
    images.extend([b,heatmap,res])
    grid_image=make_grid(images,nrow=3)
    imshow2(grid_image,mis_class[index-1]['pred'],mis_class[index - 1]['target'])