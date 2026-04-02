#include <iostream>
#include "engine.hpp"

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

int main() {
    manual_backprop_demo();
}