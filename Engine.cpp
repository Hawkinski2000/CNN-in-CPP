#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
#include "Engine.h"
#include "Node.h"
using namespace std;


// Function to build a topological ordering of the automatic differentiation graph
void Engine::trace_graph(shared_ptr<Node> node, unordered_set<Node*>& visited, vector<Node*>& topo) {
    if (!node || visited.count(node.get())) {
        return;
    }
    
    visited.insert(node.get());

    for (shared_ptr<Node>& child : node->children) {
        trace_graph(child, visited, topo);
    }

    topo.push_back(node.get());
}

// Function to compute the gradient of the current tensor w.r.t. graph leaves.
void Engine::run_backward(shared_ptr<Node> root) {
    unordered_set<Node*> visited;
    vector<Node*> topo;
    trace_graph(root, visited, topo);
    for (size_t i = topo.size(); i-- > 0;) {
        topo[i]->backward();
    }
}

// Function to clear the automatic differentiation graph starting from the root node.
void Engine::clear_graph(shared_ptr<Node> root) {
    unordered_set<Node*> visited;
    vector<Node*> topo;
    
    trace_graph(root, visited, topo);

    for (size_t i = 0; i < topo.size(); i++) {
        topo[i]->children.clear();
    }
}
