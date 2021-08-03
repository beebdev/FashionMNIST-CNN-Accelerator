#include <stdint.h>
#include "cnn.h"

typedef struct case_t {
  VALUE_TYPE img[28][28];
  VALUE_TYPE *output;
} case_t;

typedef struct cases_t {
  case_t *c_data;
  int32_t case_count;
} cases_t;

cases_t read_test_cases();
void free_test_cases(cases_t cases);
bool max_bin(VALUE_TYPE *expected, VALUE_TYPE *obtained);