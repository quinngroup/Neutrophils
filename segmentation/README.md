### Neutrophil Segmentation
---

This folder contains the code to segment the neutrophils from input videos. The segmented neutrophils are then used to train the generative adversarial networks (GANs). We utilized 100-Layer Tiramisu architecture for the same. The folder contains following pieces of code:

1. [tiramisu.py](https://github.com/quinngroup/Neutrophils/blob/master/segmentation/tiramisu.py): Contains the class files to train the 100-Layer tiramisu model along with the network.
2. [wborder_wholeimg_224x224.ipynb](https://github.com/quinngroup/Neutrophils/blob/master/segmentation/wborder_wholeimg_224x224.ipynb): Notebook to train the model on neutrophil images. This notebook expects the input images to be standardized and normalized. It also expects the segmentaion maps to contain the class labels directly.
3. [segmented_samples.ipynb](https://github.com/quinngroup/Neutrophils/blob/master/segmentation/segmented_samples.ipynb): Displays few results of segmentation.

#### Usage

Just follow the steps in [wborder_wholeimg_224x224.ipynb](https://github.com/quinngroup/Neutrophils/blob/master/segmentation/wborder_wholeimg_224x224.ipynb) notebook.

---
### Acknowledgments
1. [Fast.ai](https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb)
2. [There's Waldo!](https://github.com/bckenstler/TheresWaldo)
3. Jégou, S., Drozdzal, M., Vázquez, D., Romero, A., and Bengio, Y. (2016). The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation. CoRR, abs/1611.09326.
