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
1. Compile the cnn-training program in the ```cnn_training``` directory by running ```make all``` in the directory.
2. Run the program to obtain a file of parameters that will be saved to ```cnn_accelerator/include/weights.h```
3. Compile the FasionMNIST-cnn code in the  ```cnn_accelerator``` directory
4. Run ```fasionMNIST``` to execute image recognition on the test dataset
5. Run the python script to obtain results.

## Resources
* Fashion-MNIST repository: https://github.com/zalandoresearch/fashion-mnist
* Tensorflow keras tutorial on classifying FasionMNIST: https://www.tensorflow.org/tutorials/keras/classification
* simple-cnn repository: https://github.com/can1357/simple_cnn