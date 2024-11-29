# Car detection project

## Description
This project is an introduction to object recognition in python using PyTorch. <br>
It is a mean for me to put my foot in the computer vision field and apply techniques learnt
through the various online courses I took, and some college courses I follow.
I am also using this project to start reading more scientific papers, and implement some solutions into this project.<br>

## Files
- <b>[detection.ipynb]</b> is using a faster RCNN pre-trained model to detect cars on our kaggle dataset. <br>
- <b>[own-model.ipynb]</b> aims to create our own model to detect those cars. For now it is a CNN, that we update frequently and that we evaluate on an IoU score.<br>
This is an <b>ONGOING</b> project, the notebook is not well organized yet. (see kaggle repository to see more organized code)<br>
<br>

## Early Results
Our latest model reached at its best around 0.14 IoU, generating such prediction on the test set :<br>
[![20JAC4j.md.png](https://iili.io/20JAC4j.md.png)](https://freeimage.host/i/20JAC4j)

But also generating bounding boxes like this one :<br>
[![20JTOVs.md.png](https://iili.io/20JTOVs.md.png)](https://freeimage.host/i/20JTOVs)

## Ameliorations
Non-exhaustive list of ameliorations hints :
- Implementing GIoU loss function (and perhaps metric) from [Rezatofighi et al.](https://giou.stanford.edu/GIoU.pdf) (2019)
- Testing other loss functions (D-IoU, L1-smooth)
- Testing asymetrical kernel sizes and deeper models

## References
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box
Regression](https://giou.stanford.edu/GIoU.pdf) <br>
- [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://cdn.aaai.org/ojs/6999/6999-13-10228-1-10-20200525.pdf)
- Classic Computer Vision papers (LeNet, ALexNet,...)

## Data
https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data
