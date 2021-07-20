#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include "include/utils.h"
#include "include/cnn.h"

std::string logo = "\u001b[34;1m╔═╗┌─┐┌─┐┬ ┬┬┌─┐┌┐┌╔═╗╔╗╔╔╗╔  \n\
╠╣ ├─┤└─┐├─┤││ ││││║  ║║║║║║ \n\
╚  ┴ ┴└─┘┴ ┴┴└─┘┘└┘╚═╝╝╚╝╝╚╝ \n\
╔═╗┌─┐┌─┐┌─┐┬  ┌─┐┬─┐┌─┐┌┬┐┌─┐┬─┐ \n\
╠═╣│  │  ├┤ │  ├┤ ├┬┘├─┤ │ │ │├┬┘ \n\
╩ ╩└─┘└─┘└─┘┴─┘└─┘┴└─┴ ┴ ┴ └─┘┴└─\n\u001b[0m";

int main() {
    std::cout << logo << std::endl;

    /* Read test cases */
    cases_t cases = read_test_cases();

    /* Run classification on test cases */
    double total_duration = 0.0;
    std::cout << "Starting classification!" << std::endl;
    for (int c = 0; c < cases.case_count; c++) {
        /* Current test case */
        case_t curr_case = cases.c_data[c];
        float results[10];

        /* Task interval stats */
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        cnn(curr_case.img, results);
        gettimeofday(&t2, NULL);
        total_duration += (t2.tv_sec - t1.tv_sec) * 1000;
        total_duration += (t2.tv_usec - t1.tv_usec) / 1000;

        // TODO: Do stuff with the results (check accuracy)
    }

    /* Report time stats */
    std::cout << "=====================" << std::endl;
    std::cout << "Total classifcations ran: " << cases.case_count << std::endl;
    std::cout << "Total task duration (cnn task time): " << total_duration << "sec" << std::endl;
    std::cout << "Average task duration: " << total_duration / cases.case_count << "sec" << std::endl;

    /* Free cases */
    free_test_cases(cases);

    return 0;
}