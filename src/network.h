#pragma once
#include <vector>
#include "Eigen/Dense"

class Network
{
    // Ta in en lista på antal neuroner i varje lager och skapa lager utifrån
    // det. Hur gör vi med input och output neroner?
    // Initiallisera random?, Viktmatriser?
    // Biasvektor som har liknande längs som antal lager
    // En viktmatris per lager, har dimsensioner som matchar varje
    public:
    Network(std::vector<int> layers);
    Network(std::vector<Eigen::MatrixXf> w, std::vector<Eigen::VectorXf> b):weights{w},biass{b}{}
    //~Network();
    //Network(int gurk);
    //Network(Network const & other);
    //Network(Network && other);
    //Network & operator=(Network const & rhs);
    //Network & operator=(Network && rhs);

    Eigen::VectorXf compute(Eigen::VectorXf input); //Fix längd från konstruktion
    void train(std::vector<Eigen::VectorXf> input, std::vector<Eigen::VectorXf> solution, float learning_rate, unsigned int batchsize); //Fix längd från konstruktion
    void update_mini_batch(std::vector<Eigen::VectorXf> input, std::vector<Eigen::VectorXf> solution, float learning_rate); //Fix längd från konstruktion
    std::tuple<std::vector<Eigen::MatrixXf>,std::vector<Eigen::VectorXf>> gradient(Eigen::VectorXf input, Eigen::VectorXf solution);
    std::vector<Eigen::MatrixXf> weights{}; // fix längd från konstruktion
    std::vector<Eigen::VectorXf> biass{}; //Fix längd från konstruktion
};
