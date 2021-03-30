#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <string>
#include <iostream>
#include "Eigen/Dense"
#include "network.h"
#include <list>
#include <iterator>     // std::next
#include "mnist/mnist_reader.hpp"
using namespace std;
using namespace Eigen;

TEST_CASE("Hello world")
{

    CHECK(1 == 1);

    Matrix3f matris{};
    matris << 1,2,3,
              4,5,6,
              7,8,9;
    Vector3f vec{};
    vec << 1,0,0;
    Vector3f gurk{matris * vec};
    CHECK(Vector3f{1,2,3} == Vector3f(1,2,3));
    CHECK(Vector3f{1,4,7} == gurk);
    Vector3f gurkan = exp(vec.array());

    std::list<int> mylist;
    for (int i=0; i<10; i++) mylist.push_back (i*10);
    CHECK(*prev(mylist.end()) == 90);

}

TEST_CASE("MNIST")
{
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << "." << std::endl;
    int i{};
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(".");

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of pixels in first training image = " << dataset.training_images.front().size() << std::endl;
    std::cout << "Value of first pixel = " << to_string(dataset.training_images.front().front()) << std::endl;
    std::cout << "Value of first label = " << to_string(dataset.training_labels.front()) << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
}

TEST_CASE("Feedforward")
{
    //2-1-2 layer network to test if first input is larger than second input.
    //The brightest one should light up for example input [0.5, 8] gives [0,1]
    //I make up resonable weights

    //Size check when creating neural network with premade weights?
    MatrixXf w1(2,2);
    w1 << 10000,-10000,
          -10000,10000;
    //MatrixXf w2(2,1);
    //w2 << 1,
          //2;
    VectorXf b1(2);
    b1 << 0,
          0;
    //VectorXf b2(2);
    //b2 << 1,3;

    std::cout << "w1 = " << w1 << std::endl;
    //std::cout << "w2 = " << w2 << std::endl;
    std::cout << "b1 = " << b1 << std::endl;
    //std::cout << "b2 = " << b2 << std::endl;

    vector<VectorXf> b_s{b1};
    vector<MatrixXf> w_s{w1};

    Network network{w_s,b_s};
    CHECK(network.compute(Vector2f(5,6))== Vector2f(0,1));
    CHECK(network.compute(Vector2f(8,6))== Vector2f(1,0));
    CHECK(network.compute(Vector2f(9,9))== Vector2f(0.5,0.5));

    Network network2{vector<int>{3,10,7,2,8,3}};

    VectorXf input(3);
    input << 0.2,
          0.3,
          0.9;

    std::cout << "input = " << input << std::endl;
    Vector3f output = network2.compute(input);
    std::cout << "output = " << output << std::endl;


    MatrixXf w2(1,1);
    w2 << 2;
    VectorXf b2(1);
    b2 << 3;

    vector<VectorXf> b_s2{b2};
    vector<MatrixXf> w_s2{w2};

    Network network3{w_s2,b_s2};
    VectorXf input3(1);
    input3 << 0.1;
    VectorXf result3(1);
    result3 << 0.9608342772;
    CHECK(network3.compute(input3) == result3);

}

TEST_CASE("Backprop")
{
    //Ide:
    //
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
