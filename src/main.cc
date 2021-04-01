//#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "network.h"
//#include <stack>
#include "Eigen/Dense"
#include "mnist/mnist_reader.hpp"

using namespace std;
using namespace Eigen;

//returnera slumpat par av en vector med slumpade tal och en vector som visar
//rätt tal.
std::tuple<Eigen::Vector2f, Eigen::Vector2f> datapoint()
{
    Vector2f x = Vector2f::Random();
    Vector2f y{};
    if(x[0] == x[1])
    {
        y = Vector2f{0.5,0.5};
    }
    else if(x[0] < x[1])
    {
        y = Vector2f{0,1};
    }
    else
    {
        y = Vector2f{1,0};

    }
    return std::make_tuple(x,y);
}


int main()
{
    //training data
    int n_train{1000000};
    unsigned n_test{1000};
    Vector2f tmp_input{};
    Vector2f tmp_sol{};
    //testing data
    vector<VectorXf> t_input{};
    vector<VectorXf> t_solution{};
    vector<VectorXf> t_output{};
    vector<VectorXf> input{};
    vector<VectorXf> solution{};
/*
    VectorXf xfgas(1);
    VectorXf xfgas2(10);
    xfgas << 1.8;
    xfgas2 << 0,0,0,1,0,0,0,0,0,0;
    vector<VectorXf> input(10000, VectorXf(xfgas));
    vector<VectorXf> solution(10000,VectorXf(xfgas2));
    */

    /*
    for(int i{0}; i < n_train; i++)
    {
        tie(tmp_input, tmp_sol) = datapoint();
        input.push_back(tmp_input);
        solution.push_back(tmp_sol);
    }


    for(unsigned i{0}; i < n_test; i++)
    {
        tie(tmp_input, tmp_sol) = datapoint();
        t_input.push_back(tmp_input);
        t_solution.push_back(tmp_sol);
    }
    */


    mnist::MNIST_dataset<std::vector, std::vector<float>, u_int8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, float, uint8_t>(".");
    //Konvertera till vector<VectorXf>
    for(vector<float> u: dataset.training_images)
    {
        VectorXf gurk(u.size());
        for(unsigned n{0}; n < u.size(); n++)
        {
            gurk[n] = u[n];
        }
        input.push_back(gurk);
    }
    VectorXf v(10);
    for(u_int8_t u: dataset.training_labels)
    {
        for(unsigned i{0}; i <10;i++)
        {
            v[i] = u == i ? 1 : 0;
        }
        solution.push_back(VectorXf{v});
    }

    for(vector<float> u: dataset.test_images)
    {
        VectorXf gurk(u.size());
        for(unsigned n{0}; n < u.size(); n++)
        {
            gurk[n] = u[n];
        }
        t_input.push_back(gurk);
    }
    for(u_int8_t u: dataset.test_labels)
    {
        for(unsigned i{0}; i <10;i++)
        {
            v[i] = u == i ? 1 : 0;
        }
        t_solution.push_back(VectorXf{v});
    }

    //Network with this type of layer
    Network network{vector<int>{784,100,30,10}};
    network.train(input, solution, 3, 10);

    for(unsigned i{0}; i < t_input.size(); i++)
    {
        t_output.push_back(network.compute(t_input[i], true));
    }

    int n_correct{0};// Hur mäter vi det här igentligen? Någon form av avstånd?

    for(unsigned i{0}; i < t_output.size(); i++)
    {
        if (t_output[i] == t_solution[i])
        {
            n_correct++;
        }
    }
cout << "antal korrekta : " << n_correct << " / " << t_output.size() << "." << endl;
    // genera data
    // träna
    // generera testdata
    // beräkna testdata
    // utvärdera testdata

}

