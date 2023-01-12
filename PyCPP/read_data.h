#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::Matrix, Eigen::Dynamic;

typedef Matrix<double, Eigen::Dynamic, 784> MnistTrainType;
typedef Matrix<double, Eigen::Dynamic, 1> MnistLabelsType;

class ReadData {

public:

    // Need to get access to a  N X 28 X 28 array, and the equivalent N X 10 array
    // In order to pass it onto the training model

    ReadData() {}

    ReadData(MnistTrainType imageData, MnistLabelsType labels) {
        std::cout << "First pixel data " << imageData(0, 0) << std::endl;
        std::cout << "First label data " << labels(0) << std::endl; 
    }
    
};