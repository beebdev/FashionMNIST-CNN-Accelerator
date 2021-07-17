#include "include/byteswap.h"
#include "include/cnn.h"
#include "include/types.h"
#include <string>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace std;

/* CNN one epoch of training */
float train(vector<layer_t *> &layers, tensor_t<float> &data, tensor_t<float> &expected) {
    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            activate(layers[i], data);
        } else {
            activate(layers[i], layers[i - 1]->out);
        }
    }

    tensor_t<float> grads = layers.back()->out - expected;
    for (int i = layers.size() - 1; i >= 0; i--) {
        if (i == layers.size() - 1) {
            calc_grads(layers[i], grads);
        } else {
            calc_grads(layers[i], layers[i + 1]->grads_in);
        }
    }

    for (int i = 0; i < layers.size(); i++) {
        fix_weights(layers[i]);
    }

    float err = 0;
    for (int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++) {
        float f = expected.data[i];
        if (f > 0.5) {
            err += abs(grads.data[i]);
        }
    }
    return err * 100;
}

struct case_t {
    tensor_t<float> data;
    tensor_t<float> out;
};

/* Read single file into a buffer */
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

/* Read training test cases */
vector<case_t> read_test_cases() {
    vector<case_t> cases;

    uint8_t *train_image = read_file("../data/Fashion/train-images-idx3-ubyte");
    uint8_t *train_labels = read_file("../data/Fashion/train-labels-idx1-ubyte");

    uint32_t case_count = byteswap_uint32(*(uint32_t *) (train_image + 4));

    for (int i = 0; i < case_count; i++) {
        case_t c{tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1)};

        uint8_t *img = train_image + 16 + i * (28 * 28);
        uint8_t *label = train_labels + 8 + i;

        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                c.data(x, y, 0) = img[x + y * 28] / 255.f;

        for (int b = 0; b < 10; b++)
            c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

        cases.push_back(c);
    }
    delete[] train_image;
    delete[] train_labels;

    return cases;
}


//////////////////////////////
vector<layer_t *> config_model(vector<case_t> &cases) {
    /* Setup layers for model */
    vector<layer_t *> layers;
    conv_layer_t *layer1 = new conv_layer_t(1, 5, 8, cases[0].data.size);   // 28 * 28 * 1 -> 24 * 24 * 8
    relu_layer_t *layer2 = new relu_layer_t(layer1->out.size);
    pool_layer_t *layer3 = new pool_layer_t(2, 2, layer2->out.size);        // 24 * 24 * 8 -> 12 * 12 * 8
    cout << layer3->out.size.x << " ";
    cout << layer3->out.size.y << " ";
    cout << layer3->out.size.z << " " << endl;
    fc_layer_t *layer4 = new fc_layer_t(layer3->out.size, 10);              // 4 * 4 * 16 -> 10
    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);
    return layers;
}

void model_fit(vector<layer_t *> &layers, vector<case_t> &cases, long epoch) {
    struct timeval training_t1, training_t2;
    cout << "Training start. Total epoch=" << epoch << endl;
    gettimeofday(&training_t1, NULL);

    int ic = 0;
    float amse = 0;
    for (long ep = 0; ep < epoch;) {
        for (case_t &t : cases) {
            float xerr = train(layers, t.data, t.out);
            amse += xerr;
            ep++;
            ic++;
            if (ep % 1000 == 0)
                cout << "case " << ep << " err=" << amse / ic << endl;
        }
    }
    gettimeofday(&training_t2, NULL);
    double elapsedTime = (training_t2.tv_sec - training_t1.tv_sec);
    elapsedTime += (training_t2.tv_usec - training_t1.tv_usec) / 1000000.0;
    cout << "Training done. Duration: " << elapsedTime << "sec" << endl;
}

void save_parameters(vector<layer_t *> &layers, string filepath) {
    /* Open file */
    ofstream outfile;
    outfile.open(filepath, ios::out | ios::trunc);

    /* Start saving parameters */
    for (int i = 0; i < layers.size(); i++) {
        save_weights(layers[i], outfile);
    }
    outfile.close();
}

int main() {
    /* Read test cases */
    vector<case_t> cases = read_test_cases();
    vector<layer_t *> layers = config_model(cases);
    // model_fit(layers, cases, 100000);
    model_fit(layers, cases, 1000);
    save_parameters(layers, "weights.txt");
    cout << "getParameters done." << endl;
    return 0;
}
