#pragma once
#include "engine.hpp"
#include <vector>
#include <random>

inline void zero_grad(Graph& g, const std::vector<Value>& params) {
    for (Value p : params) g.nodes[p].grad = 0.0;
}

inline void sgd_step(Graph& g, const std::vector<Value>& params, double lr) {
    for (Value p : params) g.nodes[p].data -= lr * g.nodes[p].grad;
}


struct Neuron {
    std::vector<Value> w;
    Value b;
    bool nonlin;

    Neuron(Graph &g, size_t nin, bool nonlin, std::mt19937& rng)
        : b(g.value(0.0)), nonlin(nonlin)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        w.reserve(nin);
        for(size_t i = 0; i < nin; ++i) {
            w.push_back(g.value(dist(rng)));
        }
    }
};