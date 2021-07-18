#include <stdint.h>
#include <cfloat.h>

void cnn(float **img, float *result);
void conv_layer();
void relu_layer();
void pool_layer();
void fc_layer();