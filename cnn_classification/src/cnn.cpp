#include "cnn.h"
#include "weights.h"
#include <math.h>
#include <float.h>


/* CONV */
float kernel_conv(float din[CONV_IN_DIM_X][CONV_IN_DIM_Y], int x, int y, int filter) {
    /* Map x, y */
    int o_x = x * CONV_STRIDE;
    int o_y = y * CONV_STRIDE;

    float sum = 0;
    for (int i = 0; i < CONV_EXTFILTER; i++) {
        for (int j = 0; j < CONV_EXTFILTER; j++) {
            float f = conv_filter[filter][i][j];
            float v = din[o_x + i][o_y + j];
            sum += f * v;
        }
    }
    return sum;
}

/* Conv_layer
 * din: 2D array of 28*28
 * dout: 3D array of */
void conv_layer(float din[CONV_IN_DIM_X][CONV_IN_DIM_Y], float dout[CONV_OUT_DIM_X][CONV_OUT_DIM_Y][CONV_OUT_DIM_Z]) {
    for (int filter = 0; filter < CONV_NFILTERS; filter++) {
        for (int x = 0; x < CONV_OUT_DIM_X; x++) {
            for (int y = 0; y < CONV_OUT_DIM_Y; y++) {
                // Call kernel convolution
                dout[x][y][filter] = kernel_conv(din, x, y, filter);
            }
        }
    }
}


void relu_layer(float din[CONV_OUT_DIM_X][CONV_OUT_DIM_Y][CONV_OUT_DIM_Z], float dout[RELU_DIM_X][RELU_DIM_Y][RELU_DIM_Z]) {
    for (int x = 0; x < RELU_DIM_X; x++) {
        for (int y = 0; y < RELU_DIM_Y; y++) {
            for (int z = 0; z < RELU_DIM_Z; z++) {
                float v = din[x][y][z];
                if (v < 0) {
                    v = 0;
                }
                dout[x][y][z] = v;
            }
        }
    }
}

/* Pool_layer
 * din: 2D array of 24 * 24
 * dout: 3D array of 12 * 12 * 8 */
void pool_layer(float din[RELU_DIM_X][RELU_DIM_Y][RELU_DIM_Z], float dout[POOL_OUT_DIM_X][POOL_OUT_DIM_Y][POOL_OUT_DIM_Z]) {
    // TODO: implement me
    for (int x = 0; x < POOL_OUT_DIM_X; x++) {
        for (int y = 0; y < POOL_OUT_DIM_Y; y++) {
            for (int z = 0; z < POOL_OUT_DIM_Z; z++) {
                float mval = -FLT_MAX;
                int stride_x = x * POOL_STRIDE;
                int stride_y = y * POOL_STRIDE;
                for (int i = 0; i < POOL_EXTFILTER; ++i) {
                    for (int j = 0; j < POOL_EXTFILTER; ++j) {
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
int map(float dz, float dy, float dx) {
    return dz * (FC_IN_DIM_X * FC_IN_DIM_Y) + dy * (FC_IN_DIM_X) +dx;
}

float activator_function(float x) {
    // TODO: Might needa consider changing this to relu (easier)
    float sig = 1.0f / (1.0f + exp(-x));
    return sig;
    // return x >= 0 ? 1 : 0; // Not as good (very bad)
}

/* Fc_layer
 * din: 3D array of 12 * 12 * 8
 * dout: 1D array of 10 */
void fc_layer(float din[POOL_OUT_DIM_X][POOL_OUT_DIM_Y][POOL_OUT_DIM_Z], float *dout) {
    for (int n = 0; n < FC_OUT_DIM_X; n++) {
        float inputv = 0;
        for (int i = 0; i < FC_IN_DIM_X; i++) {
            for (int j = 0; j < FC_IN_DIM_Y; j++) {
                for (int z = 0; z < FC_IN_DIM_Z; z++) {
                    int m = map(z, j, i);
                    inputv += din[i][j][z] * fc_weights[m][n];
                }
            }
        }
        dout[n] = activator_function(inputv);
    }
}

void cnn(float img[28][28], float result[10]) {
    float layer1_out[CONV_OUT_DIM_X][CONV_OUT_DIM_Y][CONV_OUT_DIM_Z] = { 0 };
    float layer2_out[RELU_DIM_X][RELU_DIM_Y][RELU_DIM_Z] = { 0 };
    float layer3_out[POOL_OUT_DIM_X][POOL_OUT_DIM_Y][POOL_OUT_DIM_Z] = { 0 };

    conv_layer(img, layer1_out);
    relu_layer(layer1_out, layer2_out);
    pool_layer(layer2_out, layer3_out);
    fc_layer(layer3_out, result);
}