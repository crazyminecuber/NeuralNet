#pragma once
#include <vector>
#include "Eigen/Dense"
using namespace Eigen;
class Network
{
    // Ta in en lista på antal neuroner i varje lager och skapa lager utifrån
    // det. Hur gör vi med input och output neroner?
    // Initiallisera random?, Viktmatriser?
    // Biasvektor som har liknande längs som antal lager
    // En viktmatris per lager, har dimsensioner som matchar varje
    public:
    Network(std::vector<int> layers);
    Network(std::vector<MatrixXf> w, std::vector<VectorXf> b):weights{w},biass{b}{}
    //~Network();
    //Network(int gurk);
    //Network(Network const & other);
    //Network(Network && other);
    //Network & operator=(Network const & rhs);
    //Network & operator=(Network && rhs);


    VectorXf compute(VectorXf input); //Fix längd från konstruktion
    void train(VectorXf input, VectorXf solution); //Fix längd från konstruktion
    std::vector<VectorXf> biass{}; //Fix längd från konstruktion
    std::vector<MatrixXf> weights{}; // fix längd från konstruktion, :q

};
