//#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "network.h"
//#include <stack>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;


int main()
{
    MatrixXf w1(2,2);
    w1 << 1,1,
          1,1;
    MatrixXf w2(2,2);
    w2 << 1,1,
          1,1;
    VectorXf b1(2);
    b1 << 1,
          1;
    VectorXf b2(2);
    b2 << 1,
          1;

    VectorXf sol(2);
    sol << 0,
          1;

    VectorXf input(2);
    input << 1,
          2;

    std::cout << "w1 = " << w1 << std::endl;
    std::cout << "w2 = " << w2 << std::endl;
    std::cout << "b1 = " << b1 << std::endl;
    std::cout << "b2 = " << b2 << std::endl;
    std::cout << "input = " << input << std::endl;
    std::cout << "sol = " << sol << std::endl;

    vector<VectorXf> nabla_b{};
    vector<MatrixXf> nabla_w{};

    vector<VectorXf> b_s{b1,b2};
    vector<MatrixXf> w_s{w1,w2};

    Network network{w_s,b_s};

    tie(nabla_w, nabla_b) = network.gradient(input, sol);

    std::cout << "nabla_w.size() = " << nabla_w.size() << std::endl;
    std::cout << "nabla_b.size() = " << nabla_b.size() << std::endl;
    std::cout << "nabla_b.front() = " << nabla_b.front() << std::endl;
    std::cout << "nabla_w.front() = " << nabla_w.front() << std::endl;
    std::cout << "nabla_b.back() = " << nabla_b.back() << std::endl;
    std::cout << "nabla_w.back() = " << nabla_w.back() << std::endl;
}

