#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
// #include <ppl.h>

#define FILTER_SIZE 3
#define ARR_3D vector<vector<vector<int>>>
// force to reallocate the used memory, not sure if this works
#define DELETE_VEC(data)  vector<vector<vector<int>>>().swap(data);

using namespace std;

ARR_3D initializeData(int inputHeight, int inputWidth, int inputChannels, bool isRand);

class VGG {
public:
    VGG() {}
    ARR_3D convolution(ARR_3D data, int filterNum);
    ARR_3D maxPooling(ARR_3D data, int poolSize, int strides);
    void fullyConnected(int inputHeight, int inputWidth, int inputChannels, int neuralNum);
    void fullyConnected(int prevNeuralNum, int neuralNum);

private:
    int activation(int input);
    ARR_3D zeroPadding(ARR_3D data);
};

ARR_3D VGG::convolution(ARR_3D data, int filterNum)
{
    int padding = 1;
    int stride = 1;
    int batchSize = 1;

    int inputHeight = data.size();
    int inputWidth = data[0].size();
    int inputChannels = data[0][0].size();
    int outputHeight = (inputHeight - FILTER_SIZE + 2 * padding - stride) / stride;
    int outputWidth = (inputWidth - FILTER_SIZE + 2 * padding - stride) / stride;

    int memSize = inputHeight * inputWidth * inputChannels;
    // filterNum is number of channels for the next layer
    int paraNum = FILTER_SIZE * FILTER_SIZE * inputChannels * filterNum + filterNum;
    int macNum = inputHeight * inputWidth * FILTER_SIZE * FILTER_SIZE * filterNum * batchSize;

    printf("=========================[Conv]=========================\n"
            "              Input: %d x %d x %d\n"
            "             Output: %d x %d x %d\n"
            "        Memory size: %d x %d x %d = %d\n"
            "     # of parameter: %d x %d x %d x %d + %d = %d\n"
            "# of MAC operations: %d x %d x %d x %d x %d x %d = %d\n\n", 
            inputHeight, inputWidth, inputChannels,
            outputHeight + 2, outputWidth + 2, filterNum,
            inputHeight, inputWidth, inputChannels, memSize, 
            FILTER_SIZE, FILTER_SIZE, inputChannels, filterNum, filterNum, paraNum,
            inputHeight, inputWidth, FILTER_SIZE, FILTER_SIZE, filterNum, batchSize, macNum);

    ARR_3D Y = initializeData(outputHeight, outputWidth, filterNum, false);
    ARR_3D filter = initializeData(FILTER_SIZE, FILTER_SIZE, inputChannels, true); 

    for (int m = 0; m < filterNum; ++m) {
        for (int x = 0; x < outputHeight; ++x) {
            for (int y = 0; y < outputWidth; ++y) {
                // for each output feature map value, convolute and activate
                for (int i = 0; i < FILTER_SIZE; ++i) { 
                    for (int j = 0; j < FILTER_SIZE; ++j) {
                        for (int k = 0; k < inputChannels; ++k) {
                            // MAC
                            Y[x][y][m] += data[x+i][y+j][k] * filter[i][j][k];
                        }
                    }
                }
                Y[x][y][m] = this->activation(Y[x][y][m]);
            }
        }
    }
    
    DELETE_VEC(data)
    return this->zeroPadding(Y);
}

ARR_3D VGG::maxPooling(ARR_3D data, int poolSize, int strides)
{
    int inputHeight = data.size();
    int inputWidth = data[0].size();
    int inputChannels = data[0][0].size();
    int outputSize = (inputHeight - poolSize) / strides + 1;

    // TODO number of MAC ???
    printf("=========================[Pool]=========================\n"
            "              Input: %d x %d x %d\n"
            "             Output: %d x %d x %d\n"
            "# of MAC operations: 0\n\n",
            inputHeight, inputWidth, inputChannels,
            outputSize, outputSize, inputChannels);
    
    ARR_3D Y = initializeData(outputSize, outputSize, inputChannels, false);

    for (int m = 0; m < inputChannels; ++m) {
        for (int x = 0; x < outputSize; ++x) {
            for (int y = 0; y < outputSize; ++y) {
                // for each pooled value
                int maxVal = -2147483648;
                for (int i = 0; i < poolSize; ++i) { 
                    for (int j = 0; j < poolSize; ++j) {
                        if (data[strides*x + i][strides*y + j][m] > maxVal)
                            maxVal = data[strides*x + i][strides*y + j][m];
                    }
                }
                Y[x][y][m] = maxVal;
            }
        }
    }

    DELETE_VEC(data)
    return Y;
}

// if the previous layer of the FC layer is a Conv layer
void VGG::fullyConnected(int inputHeight, int inputWidth, int inputChannels, int neuralNum)
{
    int memSize = inputHeight * inputWidth * inputChannels;
    int paraNum = inputHeight * inputWidth * inputChannels * neuralNum;
    int macNum = inputHeight * inputWidth * FILTER_SIZE * neuralNum;

    printf("[FC] Input: %d x %d x %d  Output: %d\n"
            "        Memory size: %d x %d x %d = %d\n"
            "     # of parameter: %d x %d x %d x %d + %d = %d\n"
            "# of MAC operations: %d x %d x %d x %d = %d\n\n", 
            inputHeight, inputWidth, inputChannels, neuralNum,
            inputHeight, inputWidth, inputChannels, memSize, 
            inputHeight, inputWidth, inputChannels, neuralNum, neuralNum, paraNum,
            inputHeight, inputWidth, FILTER_SIZE, neuralNum, macNum);

}

// if the previous layer of the FC layer is a FC layer
void VGG::fullyConnected(int prevNeuralNum, int neuralNum)
{
    int paraNum = prevNeuralNum * neuralNum;
    int macNum = prevNeuralNum * neuralNum;

    printf("[FC] Input: %d  Output: %d\n"
            "        Memory size: %d\n"
            "     # of parameter: %d x %d = %d\n"
            "# of MAC operations: %d x %d = %d\n\n", 
            prevNeuralNum, neuralNum,
            prevNeuralNum,
            prevNeuralNum, neuralNum, paraNum,
            prevNeuralNum, neuralNum, macNum);
}

int VGG::activation(int input)
{
    // implement ReLU function: x = max(0, x)
    return (input > 0) ? input : 0; 
}

ARR_3D VGG::zeroPadding(ARR_3D data)
{
    int h = data.size() + 2;
    int w = data[0].size() + 2;
    ARR_3D padData = initializeData(h, w, data[0][0].size(), false);
    
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int k = 0; k < data[0][0].size(); ++k) {
                if (i != 0 && i != h-1 && j != 0 && j != w-1) {
                    padData[i][j][k] = data[i-1][j-1][k];
                }
            }
        }
    }

    DELETE_VEC(data)
    return padData;
}

ARR_3D initializeData(int inputHeight, int inputWidth, int inputChannels, bool isRand)
{
    // seed
    srand(time(NULL));

    // data will be inputHeight x inputWidth x inputChannels
    ARR_3D data;
    for (int i = 0; i < inputHeight; ++i) {
        data.push_back({});
        for (int j = 0; j < inputWidth; ++j) {
            data[i].push_back({});
            for (int k = 0; k < inputChannels; ++k) {
                int x = isRand ? rand() % 10 : 0;
                data[i][j].push_back(x);
            }
        }
    }

    return data;
}

int main() 
{
    VGG net;
    ARR_3D data = initializeData(224, 224, 3, true);
    ARR_3D y1 = net.convolution(data, 64);
    y1 = net.convolution(y1, 64);
    ARR_3D y2 = net.maxPooling(y1, 2, 2);
    ARR_3D y3 = net.convolution(y2, 128);
    y3 = net.convolution(y3, 128);
    ARR_3D y4 = net.maxPooling(y3, 2, 2);
    ARR_3D y5 = net.convolution(y4, 256);
    y5 = net.convolution(y5, 256);
    y5 = net.convolution(y5,  256);
    ARR_3D y6 = net.maxPooling(y5, 2, 2);
    ARR_3D y7 = net.convolution(y6, 512);
    y7 = net.convolution(y7, 512);
    y7 = net.convolution(y7, 512);
    ARR_3D y8 = net.maxPooling(y7, 2, 2);
    y8 = net.convolution(y8, 512);
    y8 = net.convolution(y8, 512);
    y8 = net.convolution(y8, 512);
    ARR_3D y9 = net.maxPooling(y8, 2, 2);
    // net.fullyConnected(7, 7, 512, 4096);
    // net.fullyConnected(4096, 4096);
    // net.fullyConnected(4096, 1000);

    return 0;
}
