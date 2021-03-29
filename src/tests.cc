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

    std::cout << "hi" << std::endl;
    Network network{w_s,b_s};
    std::cout << "hi2" << std::endl;
    VectorXf input(2);
    input << 2,
             1;
    VectorXf output = network.compute(input);

    std::cout << "input = " << input << std::endl;
    std::cout << "output = " << output << std::endl;


}

