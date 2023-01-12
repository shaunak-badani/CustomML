#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::Matrix, Eigen::Dynamic;

typedef Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> myMatrix;

class MyClass {
    int N;
    double a;
    double b;

public:

    Eigen::VectorXd v_data;
    Eigen::VectorXd v_gamma;
    
    MyClass() {}

    MyClass( double a_in, double b_in, double N_in ) 
    {
        N = N_in;
        a = a_in;
        b = b_in;
    }

    void run() 
    {
        v_data = Eigen::VectorXd::LinSpaced(N, a, b);
        auto gammafn = [](double it) { return std::tgamma(it); };
        v_gamma = v_data.unaryExpr(gammafn);
        std::cout << "Called gamma function" << std::endl;
    }

};