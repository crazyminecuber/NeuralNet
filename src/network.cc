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
    return 2 * (output_activations - sol);
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
    auto w = weights.begin();
    for (;w != weights.end() && b != biass.end(); ++w, ++b)
    {
        input = sigmoid(((*w) * input + (*b)));
    }
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

void Network::train(std::vector<Eigen::VectorXf> input, std::vector<Eigen::VectorXf> solution, float learning_rate, unsigned int batchsize)
{
    unsigned int index{0};
    while(index + batchsize < input.size())
    {
        update_mini_batch(
        std::vector<Eigen::VectorXf>(input.begin() + index,input.begin() + index + batchsize),
        std::vector<Eigen::VectorXf>(solution.begin() + index, solution.begin() + index + batchsize), // Dubbelräknas ändpunkterna?
        learning_rate
        );
        index += batchsize;
        cout << "epoc " << index / batchsize  << "done.";
    }
    update_mini_batch(
    std::vector<Eigen::VectorXf>(input.begin() + index,input.end()), // blir detta rätt?
    std::vector<Eigen::VectorXf>(solution.begin() + index, solution.end()),
    learning_rate
    );
}

void Network::update_mini_batch(std::vector<Eigen::VectorXf> input, std::vector<Eigen::VectorXf> solution, float learning_rate)
{
    vector<VectorXf> nabla_b{};
    vector<MatrixXf> nabla_w{};
    vector<VectorXf> del_nabla_b{};
    vector<MatrixXf> del_nabla_w{};

    for(unsigned int n{0}; n < weights.size(); n++)
    {
        nabla_w.push_back(MatrixXf::Zero(weights[n].rows(), weights[n].cols()));
    }
    for(unsigned int n{0}; n < biass.size(); n++)
    {
        nabla_b.push_back(VectorXf::Zero(biass[n].rows(), biass[n].cols()));
    }

    auto i = input.begin();
    auto s = solution.begin();
    for (; i != input.end() && s !=solution.end(); ++i, ++s)
    {
        tie(del_nabla_w, del_nabla_b) = gradient(*i,*s);
        for(unsigned n{0}; n < nabla_w.size(); n++)
        {
            nabla_w[n] += del_nabla_w[n];
        }

        for(unsigned n{0}; n < nabla_b.size(); n++)
        {
            nabla_b[n] += del_nabla_b[n];
        }
    }

    for(unsigned n{0}; n < weights.size(); n++)
    {
        weights[n] -= (learning_rate / input.size()) * del_nabla_w[n];
    }

    for(unsigned n{0}; n < biass.size(); n++)
    {
        biass[n] -= (learning_rate / input.size()) * del_nabla_b[n];
    }

}
// Nu vill jag börja skriva kod för att kunna ladda data och träna en modell på
// något sätt. Börja med ett nätverk som skall outputa det som är störst. av två
// tal. 2-2-2 struktur kör vi med. Då vill vi ha data. Det kan jag generera
// automatiskt i detta fallet. Sedan vill jag bestämma hur många data jag ska
// träna med och hur stora batcher jag ska köra med och hur jag genomsnittar dem
// till ett totalt steg och hur jag gör i gränsfallet att vi har färra samples
// kvar än en batch.
//
// Funktion träna tar in data, batchstorlek och learningrate. Returnerar inget
// men ändrar vikterna internt.
