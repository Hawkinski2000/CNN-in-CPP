#pragma once
#include <memory>
#include <unordered_set>
#include <vector>
#include "Node.h"
using namespace std;


class Node;

// Class for executing backpropagation through the automatic differentiation graph
class Engine {
    public:
        // Function to build a topological ordering of the automatic differentiation graph
        static void trace_graph(shared_ptr<Node> node, unordered_set<Node*>& visited, vector<Node*>& topo);

        // Function to compute the gradient of the current tensor w.r.t. graph leaves.
        static void run_backward(shared_ptr<Node> root);

        // Function to clear the automatic differentiation graph starting from the root node.
        static void clear_graph(shared_ptr<Node> root);
};
