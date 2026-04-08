#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <span>

using Value = size_t;
inline constexpr Value kNull = ~Value{0};

enum class Op : uint8_t { Add, Mul, Pow, Tanh, ReLU, Sigmoid, Exp, None };

struct Node {
    double data = 0.0;
    double grad = 0.0;
    Value left = kNull;
    Value right = kNull;
    Op op = Op::None;
};

class Graph {
public:
    std::vector<Node> nodes;

    explicit Graph(size_t reserve = 4096) { nodes.reserve(reserve); }

    [[nodiscard]] Value value(double data) {
        nodes.push_back({.data = data});
        return nodes.size() - 1;
    }

    [[nodiscard]] Value add(Value a, Value b) {
        nodes.push_back({
            .data = nodes[a].data + nodes[b].data,
            .left = a,
            .right = b,
            .op = Op::Add,
        });

        return nodes.size() - 1;
    }

    [[nodiscard]] Value mul(Value a, Value b) {
        nodes.push_back({
            .data = nodes[a].data * nodes[b].data,
            .left = a,
            .right = b,
            .op = Op::Mul,
        });

        return nodes.size() - 1;
    }

    [[nodiscard]] Value sub(Value a, Value b) { return add(a, neg(b)); }
    [[nodiscard]] Value div(Value a, Value b) { return mul(a, powi(b, -1)); }

    [[nodiscard]] Value add_const(Value a, double data_b) { return add(a, value(data_b)); }
    [[nodiscard]] Value sub_const(Value a, double data_b) { return sub(a, value(data_b)); }
    [[nodiscard]] Value mul_const(Value a, double data_b) { return mul(a, value(data_b)); }
    [[nodiscard]] Value neg(Value a) { return mul_const(a, -1.0); }


    [[nodiscard]] Value powi(Value b, int e) { return powf(b, (double)e); }
    [[nodiscard]] Value powf(Value b, double e) {
        Value c = value(e);
        nodes.push_back({
            .data = std::pow(nodes[b].data, e),
            .left = b, .right = c,
            .op = Op::Pow,
        });

        return nodes.size() - 1;
    }

    [[nodiscard]] Value sigmoid(Value a) {
        nodes.push_back({
            .data = 1.0 / (1.0 + std::exp(-nodes[a].data)),
            .left = a,
            .op = Op::Sigmoid,
        });

        return nodes.size() - 1;
    }

    [[nodiscard]] Value relu(Value a) {
        nodes.push_back({
            .data = nodes[a].data > 0.0 ? nodes[a].data : 0.0,
            .left = a,
            .op   = Op::ReLU
        });
        return nodes.size() - 1;
    }

    [[nodiscard]] Value tanh(Value a) {
        nodes.push_back({
            .data = std::tanh(nodes[a].data),
            .left = a,
            .op   = Op::Tanh
        });
        return nodes.size() - 1;
    }

    [[nodiscard]] Value exp(Value a) {
        nodes.push_back({
            .data = std::exp(nodes[a].data),
            .left = a,
            .op   = Op::Exp
        });
        return nodes.size() - 1;
    }

    // BECAUSE WE USE ARENA ALLOCATOR
    // WE DON'T NEED TOPO SORT HERE
    void backward(Value root) {
        for (auto& n : nodes) n.grad = 0.0;
        nodes[root].grad = 1.0;

        for(auto i = static_cast<ptrdiff_t>(nodes.size()) - 1; i >= 0; --i) {
            const auto [data, grad, left, right, op] = nodes[i];
            
             switch (op) {
                case Op::Add:
                    nodes[left].grad  += grad;
                    nodes[right].grad += grad;
                    break;

                case Op::Mul:
                    nodes[left].grad  += nodes[right].data * grad;
                    nodes[right].grad += nodes[left].data  * grad;
                    break;

                case Op::Sigmoid:
                    nodes[left].grad += data * (1.0 - data) * grad; 
                    break;

                case Op::ReLU:
                    if (nodes[left].data > 0.0) nodes[left].grad += grad;
                    break;

                case Op::Tanh:
                    nodes[left].grad += (1.0 - data * data) * grad;
                    break;

                case Op::Exp:
                    nodes[left].grad += data * grad;
                    break;

                case Op::Pow: {
                    double e = nodes[right].data;
                    nodes[left].grad += e * std::pow(nodes[left].data, e - 1.0) * grad;
                    break;
                }

                case Op::None: break;
            }
        }
    }
};
