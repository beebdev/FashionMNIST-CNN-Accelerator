#include "include/cnn.h"
#include "include/weights.h"
#include <math.h>

void cnn(float **img, float *result) {
    // TODO: implement me
}

/* CONV */
float kernel_conv(float **din, int f, int x, int y) {
    float sum = 0;
    for (int i = 0; i < CONV_EXTFILTSIZE; i++) {
        for (int j = 0; j < CONV_EXTFILTSIZE; j++) {
            // z only has dimension of 1
            // something terrible 
            float f = conv_filter[(int)f][i][j];
            float v = din[x + i][y + j];
            sum += f * v;
        }
    }
    return sum;
}

/* Conv_layer
 * din: 2D array of 28*28
 * dout: 3D array of */
void conv_layer(float **din, float ***dout) {
    // TODO: implement me
    for (int filter = 0; filter < CONV_FILTERSIZE; filter++) {
        for (int x = 0; x < CONV_OUT_XY; x++) {
            for (int y = 0; y < CONV_OUT_XY; y++) {
                // Call kernel convolution
                dout[x][y][filter] = kernel_conv(din, filter, x, y);
            }
        }
    }
}


void relu_layer(float ***din,float ***dout) {
    // TODO: implement me
    for (int i = 0; i < RELU_IN_X; i++)
        for (int j = 0; j < RELU_IN_Y; j++)
            for (int z = 0; z < RELU_IN_Z; z++) {
                float v = din[i][j][z];
                if (v < 0)
                    v = 0;
                dout[i][j][z] = v;
            }
}

/* Pool_layer
 * din: 2D array of 24 * 24
 * dout: 3D array of 12 * 12 * 8 */
void pool_layer(float ***din, float ***dout) {
    // TODO: implement me
    for (int x = 0; x < POOL_OUT_XY; x++) {
        for (int y = 0; y < POOL_OUT_XY; y++) {
            for (int z = 0; z < POOL_OUT_Z; z++) {
                float mval = -FLT_MAX;
                int stride_x = x * POOL_STRIDE;
                int stride_y = y * POOL_STRIDE;
                for (int i = 0; i < POOL_EXTFILTSIZE; ++i) {
                    for (int j = 0; j < POOL_EXTFILTSIZE; ++j) {
                        float temp = din[stride_x + i][stride_y + j][z];
                        if (temp > mval)
                            mval = temp;
                    }
                }
                dout[x][y][z] = mval;
            }
        }
    }
}

/* FC Helper Function */
int map(float dz,float dy,float dx) {
    return dz * (FC_IN_X * FC_IN_Y) + dy * (FC_IN_X) + dx;
}

float activator_function(float x) {
    //return tanhf( x );
    float sig = 1.0f / (1.0f + exp(-x));
    return sig;
}

/* Fc_layer
 * din: 3D array of 12 * 12 * 8
 * dout: 1D array of 10 */
void fc_layer(float ***din, float *dout) {
    // TODO: implement me
    for (int n = 0; n < FC_OUT; n++) {
        float inputv = 0;

        for (int i = 0; i < FC_IN_X; i++) {
            for (int j = 0; j < FC_IN_Y; j++) {
                for (int z = 0; z < FC_IN_Z; z++) {
                    int m = map(z, j, i);
                    inputv += din[i][j][z] * fc_weights[m][n];
                }
            }
        }

        fc_inputs[n] = inputv;
        dout[n] = activator_function(inputv);
    }
}