#include <vector>
#include <iostream>
#include <deque>
#include "network.h"
#include <iterator>
#include "Eigen/Dense"
#include <tuple>

using namespace std;
using namespace Eigen;

VectorXf sigmoid(VectorXf input)
{
    return inverse(1 + exp(-input.array()));
}

VectorXf sigmoid_derivative(VectorXf z)
{
    return sigmoid(z).array()*(1 - sigmoid(z).array());
}

VectorXf cost(VectorXf output_activations, VectorXf sol)
{
    return square((output_activations - sol).array());
}

VectorXf cost_derivative(VectorXf output_activations, VectorXf sol)
{
    return 2*(output_activations - sol);
}

Network::Network(std::vector<int> layers) // Load from file somehow
{
    for (auto layer = ++layers.begin(); layer != layers.end(); ++layer)
    {
        weights.push_back(MatrixXf::Random(*layer,*prev(layer))); // Slumpad rad, kollumn. Indata avgör antal kolumner och utdata antal rader
        biass.push_back(VectorXf::Random(*layer));
    }
    std::cout << "first weight matrix" << weights.front() << std::endl;
}

VectorXf Network::compute(VectorXf input)
{
    auto b = biass.begin();
    auto w = weights.begin();
    for (;w != weights.end() && b != biass.end(); ++w, ++b)
    {
        std::cout << "different length of input" << input.size() << std::endl;
        input = sigmoid(((*w) * input + (*b)));
    }
    std::cout << "different length of input" << input.size() << std::endl;
    return input;
}

std::tuple<vector<MatrixXf>,vector<VectorXf>> Network::gradient(VectorXf input, VectorXf solution)
{
    auto b = biass.begin();
    vector<VectorXf> Z{};
    vector<VectorXf> A{input};
    deque<VectorXf> error{};

    //Feedforward
    for (auto w = weights.begin(); w != weights.end(); ++w)
    {
        Z.push_back((*w) * A.back() + (*b));
        A.push_back(sigmoid(Z.back()));
        ++b;
    }

    //Costfunction is (y-a^L)^2 so derivative is 2 * (y-a^L)
    //Calculate first layer error
    error.push_front(cost_derivative(A.back(), solution).array()*sigmoid_derivative(Z.back()).array());
    auto z = Z.rbegin();
    ++z;
    auto w = weights.rbegin();

    //backprop for all other errors
    for (; w != weights.rend() && z !=Z.rend(); ++z, ++w)
    {
        error.push_front((w->transpose() * error.front()).array() * sigmoid_derivative(*z).array());
    }

    // calculate gradients
    vector<VectorXf> nabla_b{error.begin(), error.end()};
    vector<MatrixXf> nabla_w{};
    auto e = error.begin();
    auto a = A.begin();
    for (; e != error.end() && a !=A.end(); ++e, ++a)
    {
        nabla_w.push_back((*e) * a->transpose());
    }

    return  std::make_tuple(nabla_w, nabla_b);


}



// När vi gör forward vill vi spara z i alla steg
// Vi vill ha någon funktion som returnerar gradienten för b och w
// Vi vill sedan justera vikterna på detta viset med lärningssteg eta
//
