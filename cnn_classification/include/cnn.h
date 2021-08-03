#ifndef CNN_H
#define CNN_H
#include <stdint.h>
#include "ap_fixed.h"

typedef ap_fixed<12,10> VALUE_TYPE;
//typedef float VALUE_TYPE;
void cnn(VALUE_TYPE img[28][28], VALUE_TYPE result[10]);

#endif
