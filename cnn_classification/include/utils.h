#include <stdint.h>

typedef struct case_t {
  float** img;
  float* output;
} case_t;

typedef struct cases_t {
  case_t* c_data;
  uint32_t case_count;
} cases_t;

cases_t read_test_cases();
void free_test_cases(cases_t cases);
bool max_bin(float* expected, float* obtained);