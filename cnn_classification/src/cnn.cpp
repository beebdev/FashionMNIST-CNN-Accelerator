#include "../include/cnn.h"
#include "../include/weights.h"
#include <math.h>
#include <float.h>


/* CONV */
VALUE_TYPE kernel_conv(VALUE_TYPE din[CONV_IN_DIM_X][CONV_IN_DIM_Y], int x, int y, int filter) {
    /* Map x, y */
    int o_x = x * CONV_STRIDE;
    int o_y = y * CONV_STRIDE;

    VALUE_TYPE sum = 0;
    kernel_conv_label0: for (int i = 0; i < CONV_EXTFILTER; i++) {
        kernel_conv_label1: for (int j = 0; j < CONV_EXTFILTER; j++) {
        	VALUE_TYPE f = conv_filter[filter][i][j];
        	VALUE_TYPE v = din[o_x + i][o_y + j];
            sum += f*v;
        }
    }

    if (sum < 0) {
    	return 0;
    }

    return sum;
}

/* Conv_layer
 * din: 2D array of 28*28
 * dout: 3D array of */
void conv_layer(VALUE_TYPE din[CONV_IN_DIM_X][CONV_IN_DIM_Y], VALUE_TYPE dout[CONV_OUT_DIM_X][CONV_OUT_DIM_Y][CONV_OUT_DIM_Z]) {
	conv_layer_label0: for (int filter = 0; filter < CONV_NFILTERS; filter++) {
    	conv_layer_label1: for (int x = 0; x < CONV_OUT_DIM_X; x++) {
            conv_layer_label2: for (int y = 0; y < CONV_OUT_DIM_Y; y++) {
                // Call kernel convolution
                dout[x][y][filter] = kernel_conv(din, x, y, filter);
            }
        }
    }
}


/* Pool_layer
 * din: 2D array of 24 * 24
 * dout: 3D array of 12 * 12 * 8 */
void pool_layer(VALUE_TYPE din[RELU_DIM_X][RELU_DIM_Y][RELU_DIM_Z], VALUE_TYPE dout[POOL_OUT_DIM_X][POOL_OUT_DIM_Y][POOL_OUT_DIM_Z]) {
    for (int x = 0; x < POOL_OUT_DIM_X; x++) {
        for (int y = 0; y < POOL_OUT_DIM_Y; y++) {
            pool_layer_loop_2: for (int z = 0; z < POOL_OUT_DIM_Z; z++) {
            	VALUE_TYPE mval = -FLT_MAX;
                int stride_x = x * POOL_STRIDE;
                int stride_y = y * POOL_STRIDE;
                for (int i = 0; i < POOL_EXTFILTER; ++i) {
                    for (int j = 0; j < POOL_EXTFILTER; ++j) {
                    	VALUE_TYPE temp = din[stride_x + i][stride_y + j][z];
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
int map(VALUE_TYPE dz, VALUE_TYPE dy, VALUE_TYPE dx) {
    return dz * (FC_IN_DIM_X * FC_IN_DIM_Y) + dy * (FC_IN_DIM_X) +dx;
}

VALUE_TYPE activator_function(VALUE_TYPE x) {
    // TODO: Might needa consider changing this to relu (easier)
	VALUE_TYPE sig = (VALUE_TYPE) (1.0f / (1.0f + exp((float)-x)));
    return sig;
    // return x >= 0 ? 1 : 0; // Not as good (very bad)
}

/* Fc_layer
 * din: 3D array of 12 * 12 * 8
 * dout: 1D array of 10 */
void fc_layer(VALUE_TYPE din[POOL_OUT_DIM_X][POOL_OUT_DIM_Y][POOL_OUT_DIM_Z], VALUE_TYPE dout[FC_OUT_DIM_X]) {
    
	VALUE_TYPE inputv[FC_OUT_DIM_X] = {0};
    fc_loop_3: for (int i = 0; i < FC_IN_DIM_X; i++) {
        fc_loop_2: for (int z = 0; z < FC_IN_DIM_Z; z++) {
            fc_loop_1: for (int j = 0; j < FC_IN_DIM_Y; j++) {
                int m = map(z, j, i);
                fc_loop_0 : for (int n = 0; n < FC_OUT_DIM_X; n++) {
                    inputv[n] += din[i][j][z] * fc_weights[m][n];
                }
            }
        }
    }

    fc_activation_loop: for (int n = 0; n < FC_OUT_DIM_X; n++) {
        dout[n] = activator_function(inputv[n]);
    }
}

void cnn(VALUE_TYPE img[28][28], VALUE_TYPE result[10]) {
	VALUE_TYPE layer1_out[CONV_OUT_DIM_X][CONV_OUT_DIM_Y][CONV_OUT_DIM_Z];
	VALUE_TYPE layer2_out[POOL_OUT_DIM_X][POOL_OUT_DIM_Y][POOL_OUT_DIM_Z];

    conv_layer(img, layer1_out);
    pool_layer(layer1_out, layer2_out);
    fc_layer(layer2_out, result);
}
