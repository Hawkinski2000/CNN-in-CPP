#include <iostream>
#include <chrono>
#include "Engine.h"
using namespace std;


shared_ptr<Node> Engine::root = nullptr;

size_t Engine::step = 0;
float Engine::trace_graph_total_time = 0;
float Engine::backward_loop_total_time = 0;

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
    Engine::root = root;
    unordered_set<Node*> visited;
    vector<Node*> topo;

    auto trace_graph_start = chrono::steady_clock::now();
    trace_graph(root, visited, topo);
    auto trace_graph_stop = chrono::steady_clock::now();
    auto trace_graph_duration = chrono::duration_cast<chrono::microseconds>(trace_graph_stop - trace_graph_start);
    trace_graph_total_time += trace_graph_duration.count();

    auto backward_loop_start = chrono::steady_clock::now();
    for (size_t i = topo.size(); i-- > 0;) {
        topo[i]->backward();
    }
    auto backward_loop_stop = chrono::steady_clock::now();
    auto backward_loop_duration = chrono::duration_cast<chrono::microseconds>(backward_loop_stop - backward_loop_start);
    backward_loop_total_time += backward_loop_duration.count();

    step++;
    if (step == 2810) {
        // cout << "Average trace graph duration: " << trace_graph_total_time / 2811 << " μs" << endl;
        // cout << "Average backward loop duration: " << backward_loop_total_time / 2811 << " μs" << endl;
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

// Function to get the current root node of the graph
shared_ptr<Node> Engine::get_root() {
    return root;
}
