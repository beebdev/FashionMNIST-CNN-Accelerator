#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include "../include/utils.h"
#include "../include/cnn.h"

int main() {
    /* Read test cases */
    cases_t cases = read_test_cases();

    /* Stats var */
    double total_duration = 0.0;
    long int correct = 0;

    // for (int i = 0; i < 28; i++) {
    //     for (int j = 0; j < 28; j++) {
    //         std::cout << cases.c_data[0].img[i][j] << ", " << std::ends;
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < 10; i++) {
    //     std::cout << cases.c_data[0].output[i] << ", " << std::ends;
    // }
    // std::cout << std::endl;

    /* Run classification on test cases */
    std::cout << "Starting classification!" << std::endl;
    for (int c = 0; c < cases.case_count; c++) {
        if (c % 10000 == 0 && c != 0) {
            std::cout << "case " << c;
            std::cout << " [acc: " << float(correct) / float(c) << "]" << std::endl;
        }

        /* Current test case */
        case_t curr_case = cases.c_data[c];
        VALUE_TYPE results[10];

        /* Task interval stats */
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        cnn(curr_case.img, results);
        gettimeofday(&t2, NULL);
        total_duration += (t2.tv_sec - t1.tv_sec) * 1000;
        total_duration += (t2.tv_usec - t1.tv_usec) / 1000;

        /* Current Accuracy */
        correct += max_bin(curr_case.output, results) ? 1 : 0;
    }

    /* Report time stats */
    std::cout << "=====================" << std::endl;
    std::cout << "Total classifcations ran: " << cases.case_count << std::endl;
    std::cout << "Total task duration (cnn task time): " << total_duration << "ms" << std::endl;
    std::cout << "Average task duration: " << total_duration / cases.case_count << "msec" << std::endl;
    std::cout << "Accuracy: " << float(correct) / cases.case_count << std::endl;

    /* Free cases */
    free_test_cases(cases);

    return 0;
}
