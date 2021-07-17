# FashionMNIST-CNN-Accelerator
An FPGA accelerator for CNN on the Fashion-MNIST dataset using HLS.
The aim of this project is to accelerate the classification task for image recognition on the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.
Due to the time limitation of the project duration, our model will only include:
1. CONV
2. RELU
3. POOL
4. FC
> This is a course project for COMP4601 @UNSW.

## Project Planning
This section is bound to be removed after completion of project.
https://www.notion.so/FasionMNIST-CNN-Accelerator-978da2b835f0482b9ddab8841dcc0221

## Repo structure:
* cnn_accelerator/ - Includes the C++ source code for FashionMNIST-CNN and Vivado HLS tcl script and directives.
* cnn_training/ - Includes the CNN training code that generates the parameters used in the layers in FashionMNIST-CNN
* data/ - Includes the training and test data sets for MNIST and FashionMNIST

## Quick start
### Weight training and extraction
Since we only want to accelerate the classification process, we have a seperate program for training the mdoel and saving the trained parameters to a ```weights.txt``` file. The ```getParameters.py``` script will scrap the trained weights and produce a ```weights.h``` header file saved in the ```cnn_accelerator/include```. This will be compiled to the classification program.
```bash
cd path/to/FashionMNIST-CNN-Accelerator/
cd cnn_training
./train_parameters.sh
```

### CNN Classification Accelerator

## Resources
* Fashion-MNIST repository: [link](https://github.com/zalandoresearch/fashion-mnist)
* Tensorflow keras tutorial on classifying FasionMNIST: [link](https://www.tensorflow.org/tutorials/keras/classification)
* simple-cnn repository: [link](https://github.com/can1357/simple_cnn)