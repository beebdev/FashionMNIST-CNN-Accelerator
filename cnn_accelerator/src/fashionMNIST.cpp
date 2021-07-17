#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "include/utils.h"
// #include "cnn.h"


int main() {
    /* Read test cases */
    cases_t cases = read_test_cases();
    // for (int c = 0; c < cases.case_count; c++) {
    case_t curr_case = cases.c_data[0];
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%f ", curr_case.img[i][j]);
        }
        printf("\n");
    }

    // }

    /* Free cases */
    free_test_cases(cases);

    return 0;
}