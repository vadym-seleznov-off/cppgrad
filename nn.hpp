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

    [[nodiscard]] Value forward(Graph& g, const std::vector<Value>& x) const {
        Value act = b;
        for(size_t i = 0; i < w.size(); ++i) {
            act = g.add(act, g.mul(x[i], w[i]));
        }
        return nonlin ? g.tanh(act) : act;
    }

    void collect_params(std::vector<Value>& dist) const {
        dist.insert(dist.end(), w.begin(), w.end());
        dist.push_back(b);
    }
};

struct Layer {
    std::vector<Neuron> neurons;

    Layer(Graph& g, size_t nin, size_t nout, bool nonlin, std::mt19937& rng) {
        neurons.reserve(nout);
        for(size_t i = 0; i < nout; ++i) {
            neurons.emplace_back(g, nin, nonlin, rng);
        }
    }

    [[nodiscard]] std::vector<Value> forwad(Graph& g, const std::vector<Value>& x) const {
        std::vector<Value> out;

        out.reserve(neurons.size());
        for(const auto& n : neurons) {
            out.push_back(n.forward(g, x));
        }
        return out;
    }

    void collect_params(std::vector<Value>& dist) const {
        for (const auto& n : neurons) n.collect_params(dist);
    }
};

struct MLP {
    std::vector<Layer> layers;
    std::vector<Value> params;

    MLP(Graph& g, size_t nin, const std::vector<size_t>& nouts) {
        std::mt19937 rng(std::random_device{}());
        layers.reserve(nouts.size());

        size_t prev = nin;
        for(size_t i = 0; i < nouts.size(); ++i) {
            bool nonlin = (i + 1 != nouts.size());
            layers.emplace_back(g, prev, nouts[i], nonlin, rng);
            prev = nouts[i];
        }

        for (const auto& l : layers) l.collect_params(params);
    }

    [[nodiscard]] std::vector<Value> forward(Graph& g, std::vector<Value> x) const {
        for(const auto& l : layers) {
            x = l.forwad(g, x);
        }
        
        return x;
    }
};