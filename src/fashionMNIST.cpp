#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

uint8_t*** read_mnist(const char* filepath) {
    FILE* fp;
    fp = fopen(filepath, "rb");
    if (!fp) {
        perror("Failed to open MNIST file. Exiting..");
        exit(1);
    }

    int magic_number = 0;
    int n_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // Magic num
    size_t ret = fread(&magic_number, sizeof(magic_number), 1, fp);
    if (ret != sizeof(magic_number)) {
        perror("Magic num\n");
        exit(1);
    }
    magic_number = reverse_int(magic_number);

    // num images
    ret = fread(&n_images, sizeof(n_images), 1, fp);
    if (ret != sizeof(n_images)) {
        perror("n_images\n");
        exit(1);
    }
    n_images = reverse_int(n_images);

    // num rows
    ret = fread(&n_rows, sizeof(n_rows), 1, fp);
    if (ret != sizeof(n_rows)) {
        perror("n_rows\n");
        exit(1);
    }
    n_rows = reverse_int(n_rows);

    // num cols
    ret = fread(&n_cols, sizeof(n_cols), 1, fp);
    if (ret != sizeof(n_cols)) {
        perror("n_cols\n");
        exit(1);
    }
    n_cols = reverse_int(n_cols);

    uint8_t*** buffer;
    buffer = (uint8_t***) malloc(sizeof(uint8_t**) * n_images);
    for (int i = 0; i < n_images; i++) {
        buffer[i] = (uint8_t**) malloc(sizeof(uint8_t*) * n_rows);
        for (int r = 0; r < n_rows; r++) {
            buffer[i][r] = (uint8_t*) malloc(sizeof(uint8_t) * n_cols);
            for (int c = 0; c < n_cols; c++) {
                ret = fread(&buffer[i][r][c], sizeof(uint8_t), 1, fp);
                if (ret != sizeof(uint8_t)) {
                    perror("bytes\n");
                    exit(1);
                }
            }
        }
    }

    fclose(fp);
    return buffer;
}

void read_test_cases() {
    uint8_t*** train_images = read_mnist("data/Fasion/train-images-idx3-ubyte");
    uint8_t*** train_labels = read_mnist("data/Fasion/train-labels-idx1-ubyte");

    // uint32_t case_count = byteswap_uint32(*(uint32_t*) (train_image + 4));

    // for (int i = 0; u < case_count; i++) {}
    printf("All good!\n");
    free(train_images);
    free(train_labels);
}

// void conv_layer() {
// }

// void relu_layer() {}

// void pool_layer() {}

// void fc_layer() {}

int main() {

    // Read dataset

    // Create layers

    // Training

    // Classification

    read_test_cases();
    return 0;
}