#include <iostream>
#include "engine.hpp"

int main() {
    Graph g;
    //        e
    // l = (a * b) + c

    Value a = g.value(2.0);
    Value b = g.value(-3.0);
    Value c = g.value(4.0);
    Value e = g.mul(a, b);
    Value l = g.add(e, c);

    std::cout << g.nodes[l].data << '\n';

    return 0;
}