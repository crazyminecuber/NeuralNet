#include <vector>
#include <deque>
#include "network.h"
#include <iterator>
#include "Eigen/Dense"

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

VectorXf cost(VectorXf output_activations, VectorXf a)
{
    return square((output_activations - a).array());
}

VectorXf cost_derivative(VectorXf output_activations, VectorXf a)
{
    return 2*(output_activations - a);
}

Network::Network(std::vector<int> layers) // Load from file somehow
{
    for (auto layer = ++layers.begin(); layer != layers.end(); ++layer)
    {
        weights.push_back(MatrixXf::Random(*layer,*prev(layer))); // Slumpad rad, kollumn. Indata avgör antal kolumner och utdata antal rader
        biass.push_back(VectorXf::Random(*layer));
    }
}

VectorXf Network::compute(VectorXf input)
{
    auto b = biass.begin();
    for (auto w = weights.begin(); w != weights.end(); ++w)
    {
        input = sigmoid(((*w) * input + (*b)));
        ++b;
    }
    return input;
}

void Network::train(VectorXf input, VectorXf solution)
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
}



// När vi gör forward vill vi spara z i alla steg
// Vi vill ha någon funktion som returnerar gradienten för b och w
// Vi vill sedan justera vikterna på detta viset med lärningssteg eta
//
