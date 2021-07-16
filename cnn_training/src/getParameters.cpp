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

using std::cerr;
using std::endl;
using std::ofstream;

using namespace std;

float train(vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected) {
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

void forward(vector<layer_t*>& layers, tensor_t<float>& data) {
    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            activate(layers[i], data);
        } else {
            activate(layers[i], layers[i - 1]->out);
        }
    }
}

struct case_t {
    tensor_t<float> data;
    tensor_t<float> out;
};

uint8_t* read_file(const char* szFile) {
    ifstream file(szFile, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    if (size == -1) {
        return nullptr;
    }

    uint8_t* buffer = new uint8_t[size];
    file.read((char*) buffer, size);
    return buffer;
}

vector<case_t> read_test_cases() {
    vector<case_t> cases;

    uint8_t* train_image = read_file("../data/Fashion/train-images-idx3-ubyte");
    uint8_t* train_labels = read_file("../data/Fashion/train-labels-idx1-ubyte");

    uint32_t case_count = byteswap_uint32(*(uint32_t*) (train_image + 4));

    for (int i = 0; i < case_count; i++) {
        case_t c{ tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

        uint8_t* img = train_image + 16 + i * (28 * 28);
        uint8_t* label = train_labels + 8 + i;

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

void save_tensor_float(tensor_t<float>& data,ofstream& outdata) {
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	for ( int z = 0; z < mz; z++ )
	{
        for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
                outdata << (float)data.get( x, y, z );
                outdata << " ";
            }
            outdata << "\n";
		}
	}
}

void save_layer_type(layer_type type,ofstream& outdata) {
    const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
    switch(type){
        PROCESS_VAL(layer_type::conv);     
        PROCESS_VAL(layer_type::fc);     
        PROCESS_VAL(layer_type::pool);
        PROCESS_VAL(layer_type::relu);
        PROCESS_VAL(layer_type::dropout_layer);
    }
#undef PROCESS_VAL

    outdata << s << "\n";
}

void save_integer(uint16_t val,ofstream& outdata) {
    outdata << val << "\n";
}

void save_vector_tensor_float(vector<tensor_t<float>> data,ofstream& outdata) {
    for (auto iter = data.begin(); iter != data.end(); ++iter) {
        save_tensor_float(*iter,outdata);
    } 
}

void save_tensor_gradient(tensor_t<gradient_t> data,ofstream& outdata) {
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	for ( int z = 0; z < mz; z++ )
	{
        for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
                outdata << (float)data.get( x, y, z ).grad;
                outdata << " | ";
                outdata << (float)data.get( x, y, z ).oldgrad;
                outdata << " ";
            }
            outdata << "\n";
		}
	}
}

void save_vector_tensor_gradient(vector<tensor_t<gradient_t>> data,ofstream& outdata) {
    for ( int k = 0; k < data.size(); k++ )
    {
        save_tensor_gradient(data[k],outdata);
    }
}

int main() {
    /* Read test cases */
    vector<case_t> cases = read_test_cases();
    cout << "Read data OK" << endl;

    /* Setup layers for model */
    vector<layer_t*> layers;
    conv_layer_t* layer1 = new conv_layer_t(1, 5, 8, cases[0].data.size); // 28 * 28 * 1 -> 24 * 24 * 8
    relu_layer_t* layer2 = new relu_layer_t(layer1->out.size);
    pool_layer_t* layer3 = new pool_layer_t(2, 2, layer2->out.size); // 24 * 24 * 8 -> 12 * 12 * 8
    fc_layer_t* layer4 = new fc_layer_t(layer3->out.size, 10);		 // 4 * 4 * 16 -> 10
    layers.push_back((layer_t*) layer1);
    layers.push_back((layer_t*) layer2);
    layers.push_back((layer_t*) layer3);
    layers.push_back((layer_t*) layer4);
    cout << "Layers OK" << endl;

    /* Start training */
    struct timeval training_t1, training_t2;
    cout << "Training start." << endl;
    gettimeofday(&training_t1, NULL);

    int ic = 0;
    float amse = 0;
    for (long ep = 0; ep < 200000;) {
        for (case_t& t : cases) {
            float xerr = train(layers, t.data, t.out);
            amse += xerr;
            ep++;
            ic++;
            if (ep % 1000 == 0)
                cout << "case " << ep << " err=" << amse / ic << endl;
        }
    }
    
    /* Saving Model */
    ofstream outdata; // outdata is like cin
    outdata.open("weight.txt"); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    for (int i=0; i < layers.size(); ++i) {

        save_layer_type(layers[i]->type,outdata);

        outdata << "input:\n";
        save_tensor_float(layers[i]->in,outdata);

        outdata << "output:\n";
        save_tensor_float(layers[i]->out,outdata);

        outdata << "grads_in \n";
        save_tensor_float(layers[i]->grads_in,outdata);

        //special attribute corresponding to layers
        if (layers[i]->type == layer_type::conv) {
            conv_layer_t* conv_layer = (conv_layer_t*)layers[i];
            outdata << "stride\n";
            save_integer(conv_layer->stride,outdata);
            outdata << "extend_filter\n";
            save_integer(conv_layer->extend_filter,outdata);
            outdata << "filters\n";
            save_vector_tensor_float(conv_layer->filters,outdata);
            outdata << "gradients\n";
            save_vector_tensor_gradient(conv_layer->filter_grads,outdata);
        }

    }
    outdata.close();

    // /* loading out layer */
    // vector<layer_t*> load_layers;
    // ifstream in_file;
    // in_file.open("weight.txt");

    // if (!in_file) {
    //     cerr << "Unable to open file datafile.txt";
    //     exit(1);   // call system to stop
    // }
    // string val;
    // while(in_file >> val) {
    //     if (val == "|") {
    //         continue;
    //     }
    //     if (val == "layer_type::conv") {
    //         // populate the data structure
    //         layer_t 
    //     }

    //     //cout << val << "\n";
    // }    

    /* Training done, check time duration */
    gettimeofday(&training_t2, NULL);
    double elapsedTime = (training_t2.tv_sec - training_t1.tv_sec);
    elapsedTime += (training_t2.tv_usec - training_t1.tv_usec) / 1000000.0;
    cout << "Training done. Duration: " << elapsedTime << "sec" << endl;

    
}
