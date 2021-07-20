#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "include/utils.h"

using namespace std;

uint32_t byteswap_uint32(uint32_t a) {
    return ((((a >> 24) & 0xff) << 0) |
        (((a >> 16) & 0xff) << 8) |
        (((a >> 8) & 0xff) << 16) |
        (((a >> 0) & 0xff) << 24));
}

uint8_t *read_file(const char *szFile) {
    ifstream file(szFile, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    if (size == -1) {
        return nullptr;
    }

    uint8_t *buffer = new uint8_t[size];
    file.read((char *) buffer, size);
    return buffer;
}

cases_t read_test_cases() {
    printf("Reading test cases...");

    cases_t cases;
    uint8_t *test_image = read_file("../data/Fashion/train-images-idx3-ubyte");
    uint8_t *test_labels = read_file("../data/Fashion/train-labels-idx1-ubyte");
    cases.case_count = byteswap_uint32(*(uint32_t *) (test_image + 4));
    cases.c_data = (case_t *) malloc(cases.case_count * sizeof(case_t));

    for (int c = 0; c < cases.case_count; c++) {
        /* malloc space for image data */
        cases.c_data[c].img = (float **) malloc(28 * sizeof(float *));
        for (int i = 0; i < 28; i++) {
            cases.c_data[c].img[i] = (float *) malloc(28 * sizeof(float));
        }

        /* malloc space for output data */
        cases.c_data[c].output = (float *) malloc(10 * sizeof(float));

        /* Pointer to image and label data */
        uint8_t *img = test_image + 16 + c * (28 * 28);
        uint8_t *label = test_labels + 8 + c;

        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                cases.c_data[c].img[x][y] = img[x + y * 28] / 255.f;
            }
        }

        for (int b = 0; b < 10; b++) {
            cases.c_data[c].output[b] = *label == b ? 1.0f : 0.0f;
        }

    }
    delete[] test_image;
    delete[] test_labels;
    printf("Done\n");
    return cases;
}


void free_test_cases(cases_t cases) {
    /* For each case, free images and outputs */
    for (int c = 0; c < cases.case_count; c++) {
        case_t curr_case = cases.c_data[c];

        /* Free image of a case */
        for (int i = 0; i < 28; i++) {
            free(curr_case.img[i]);
        }
        free(curr_case.img);

        /* Free output of a case */
        free(curr_case.output);
    }
}