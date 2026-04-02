#include <iostream>
#include "engine.hpp"
#include "nn.hpp"
#include <array>

void manual_backprop_demo() {
    Graph g;

    auto a = g.value( 2.0);
    auto b = g.value(-3.0);
    auto e = g.mul(a, b);           // e =  a * b  = -6
    auto c = g.value(10.0);
    auto d = g.add(e, c);           // d =  e + c  =  4
    auto f = g.value(-2.0);
    auto l = g.mul(f, d);           // l =  f * d  = -8  (output)

    g.backward(l);

    auto check = [&](Value v, double expected) {
        double got = g.nodes[v].grad;
        assert(std::abs(got - expected) < 1e-9);
    };

    check(l,  1.0);
    check(d, -2.0);
    check(f,  4.0);
    check(c, -2.0);
    check(e, -2.0);
    check(a,  6.0);
    check(b, -4.0);

    std::cout << "We are completely right!\n";
}

void train() {
    Graph g;

    // 3 inputs 2 hidden layers of size 4 and 1 output
    const std::vector<size_t> arch = {4, 4, 1};
    MLP net(g, 3, arch);

    // inputs
    constexpr double xs[4][3] = {
        { 2.0, 3.0, -1.0},
        { 3.0, -1.0, 0.5},
        { 0.5, 1.0, 1.0},
        { 1.0, 1.0, -1.0},
    };

    // desired outputs
    constexpr double ys[4] = {1.0, -1.0, -1.0, 1.0};

    const size_t params_count = net.params.size();
    double last_loss = 0.0;

    const int EPOCHS = 100;

    for(int epoch = 0; epoch < EPOCHS; ++epoch) {
        g.nodes.resize(params_count);

        // forward
        std::array<Value, 4> preds;
        for(int s = 0; s < 4; ++s) {
            std::vector<Value> x;
            x.reserve(3);
            for (double v : xs[s]) x.push_back(g.value(v));
            preds[s] = net.forward(g, std::move(x))[0];
        }

        // MSE LOSS
        Value loss = g.value(0.0);

        for (int s = 0; s < 4; ++s) {
            Value diff = g.sub(preds[s], g.value(ys[s]));
            loss = g.add(loss, g.powf(diff, 2.0));
        }

        last_loss = g.nodes[loss].data;

        // backward

        zero_grad(g, net.params);
        g.backward(loss);
        sgd_step(g, net.params, 0.01);

        std::cout << "Epoch " << epoch << ": loss " << last_loss << '\n';
    }

    assert(last_loss < 1.0 && "training did not coverge");
    std::cout << "Training passed!\n";
}

int main() {
    manual_backprop_demo();
    train();
}