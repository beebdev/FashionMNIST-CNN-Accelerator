#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include "include/utils.h"
#include "include/cnn.h"


int main() {
    /* Read test cases */
    cases_t cases = read_test_cases();

    /* Run classification on test cases */
    double total_duration;
    for (int c = 0; c < cases.case_count; c++) {
        case_t curr_case = cases.c_data[c];
        float results[10];

        /* Task interval stats */
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        cnn(curr_case.img, results);
        gettimeofday(&t2, NULL);
        total_duration += (t2.tv_sec - t1.tv_sec) * 1000;
        total_duration += (t2.tv_usec - t1.tv_usec) / 1000;
    }

    /* Report time stats */
    std::cout << "=====================" << std::endl;
    std::cout << "Total classifcations ran: " << cases.case_count << std::endl;
    std::cout << "Total task duration (only cnn time): " << total_duration << std::endl;
    std::cout << "Average task duration: " << total_duration / cases.case_count << std::endl;

    /* Free cases */
    free_test_cases(cases);

    return 0;
}