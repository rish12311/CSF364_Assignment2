#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include <cmath>
#include <limits>
#include <cassert>
#include <sstream>

using namespace std;

class Graph {
private:
    vector<unordered_set<int>> adj_list;
    unordered_set<int> vertices;
    int edge_count = 0;

public:
    void add_edge(int u, int v) {
        if (u >= adj_list.size() || v >= adj_list.size()) {
            size_t new_size = max(u, v) + 1;
            adj_list.resize(new_size);
        }
        
        if (adj_list[u].insert(v).second && adj_list[v].insert(u).second) {
            edge_count++;
            vertices.insert(u);
            vertices.insert(v);
        }
    }

    const unordered_set<int>& neighbors(int u) const {
        static const unordered_set<int> empty_set;
        return u < adj_list.size() ? adj_list[u] : empty_set;
    }

    const unordered_set<int>& get_vertices() const {
        return vertices;
    }

    int number_of_nodes() const {
        return vertices.size();
    }

    int number_of_edges() const {
        return edge_count;
    }

    bool has_edge(int u, int v) const {
        if (u >= adj_list.size() || v >= adj_list.size()) return false;
        return adj_list[u].count(v) > 0;
    }

    Graph subgraph(const unordered_set<int>& nodes) const {
        Graph sub;
        for (int u : nodes) {
            for (int v : neighbors(u)) {
                if (nodes.count(v) && u < v) {  // Avoid duplicate edges
                    sub.add_edge(u, v);
                }
            }
        }
        return sub;
    }

    void optimize_storage() {
        for (auto& neighbors : adj_list) {
            neighbors.reserve(neighbors.size());
        }
    }
};

class CoreExact {
private:
    Graph G;
    int h;
    unordered_map<int, int> clique_degrees;
    unordered_map<int, int> core_numbers;
    int kmax;
    double final_lower_bound = 0.0;
    double final_upper_bound = 0.0;

    void compute_clique_degrees() {
        clique_degrees.clear();
        vector<int> vertices(G.get_vertices().begin(), G.get_vertices().end());
        sort(vertices.begin(), vertices.end());

        unordered_map<int, int> local_counts;
        vector<int> current_clique;
        
        for (size_t i = 0; i < vertices.size(); ++i) {
            current_clique.push_back(vertices[i]);
            vector<int> candidates;
            for (size_t j = i + 1; j < vertices.size(); ++j) {
                if (G.has_edge(vertices[i], vertices[j])) {
                    candidates.push_back(vertices[j]);
                }
            }
            enumerate_cliques(current_clique, candidates, h, local_counts);
            current_clique.pop_back();
        }
        
        for (const auto& pair : local_counts) {
            clique_degrees[pair.first] += pair.second;
        }
    }

    void enumerate_cliques(vector<int>& current_clique, 
                         vector<int>& candidates,
                         int target_size,
                         unordered_map<int, int>& counts) {
        if (current_clique.size() == target_size) {
            for (int v : current_clique) counts[v]++;
            return;
        }

        for (size_t i = 0; i < candidates.size(); ++i) {
            int v = candidates[i];
            bool can_add = true;
            for (int u : current_clique) {
                if (!G.has_edge(u, v)) {
                    can_add = false;
                    break;
                }
            }
            if (can_add) {
                current_clique.push_back(v);
                vector<int> new_candidates;
                for (size_t j = i + 1; j < candidates.size(); ++j) {
                    int u = candidates[j];
                    bool connected = true;
                    for (int w : current_clique) {
                        if (!G.has_edge(w, u)) {
                            connected = false;
                            break;
                        }
                    }
                    if (connected) new_candidates.push_back(u);
                }
                enumerate_cliques(current_clique, new_candidates, target_size, counts);
                current_clique.pop_back();
            }
        }
    }

    void core_decomposition() {
        if (clique_degrees.empty()) {
            compute_clique_degrees();
        }

        core_numbers.clear();
        for (const auto& pair : clique_degrees) {
            core_numbers[pair.first] = pair.second;
        }

        int max_deg = 0;
        for (const auto& pair : clique_degrees) {
            if (pair.second > max_deg) max_deg = pair.second;
        }

        vector<vector<int>> bins(max_deg + 1);
        unordered_map<int, int> pos;

        for (const auto& pair : clique_degrees) {
            int v = pair.first;
            int deg = pair.second;
            bins[deg].push_back(v);
            pos[v] = bins[deg].size() - 1;
        }

        for (int k = 0; k <= max_deg; ++k) {
            while (!bins[k].empty()) {
                int v = bins[k].back();
                bins[k].pop_back();
                core_numbers[v] = k;

                for (int u : G.neighbors(v)) {
                    if (core_numbers.find(u) == core_numbers.end() || core_numbers[u] > k) {
                        int deg_u = clique_degrees[u];
                        if (deg_u > k) {
                            auto& bin = bins[deg_u];
                            int idx = pos[u];
                            if (idx < bin.size() && bin[idx] == u) {
                                if (idx != bin.size() - 1) {
                                    swap(bin[idx], bin.back());
                                    pos[bin[idx]] = idx;
                                }
                                bin.pop_back();
                                clique_degrees[u]--;
                                int new_deg = clique_degrees[u];
                                bins[new_deg].push_back(u);
                                pos[u] = bins[new_deg].size() - 1;
                            }
                        }
                    }
                }
            }
        }

        kmax = 0;
        for (const auto& pair : core_numbers) {
            if (pair.second > kmax) kmax = pair.second;
        }
    }

    vector<unordered_set<int>> connected_components(const Graph& graph) {
        vector<unordered_set<int>> components;
        unordered_set<int> visited;
        const auto& nodes = graph.get_vertices();

        for (int u : nodes) {
            if (visited.find(u) == visited.end()) {
                unordered_set<int> component;
                queue<int> q;
                q.push(u);
                visited.insert(u);

                while (!q.empty()) {
                    int v = q.front();
                    q.pop();
                    component.insert(v);

                    for (int w : graph.neighbors(v)) {
                        if (visited.find(w) == visited.end()) {
                            visited.insert(w);
                            q.push(w);
                        }
                    }
                }
                components.push_back(component);
            }
        }
        return components;
    }

    double compute_density(const Graph& subgraph) {
        if (subgraph.number_of_nodes() <= 1) return 0.0;

        // For h=2 (edge density), use a simple formula
        if (h == 2) {
            int n = subgraph.number_of_nodes();
            int m = subgraph.number_of_edges();
            return n > 1 ? static_cast<double>(m) / n : 0.0;
        }

        // For h > 2 (clique density), count h-cliques
        int num_cliques = 0;
        vector<int> vertices(subgraph.get_vertices().begin(), subgraph.get_vertices().end());
        sort(vertices.begin(), vertices.end());
        
        vector<int> current_clique;
        unordered_map<int, int> counts;
        
        // Start with empty clique and use all vertices as candidates
        enumerate_cliques_for_density(current_clique, vertices, h, counts, subgraph);
        
        // Count total number of h-cliques
        num_cliques = 0;
        for (const auto& pair : counts) {
            num_cliques += pair.second;
        }
        
        // Divide by h to avoid counting each clique multiple times
        return static_cast<double>(num_cliques) / h / subgraph.number_of_nodes();
    }
    
    void enumerate_cliques_for_density(vector<int>& current_clique, 
                                       vector<int>& candidates,
                                       int target_size,
                                       unordered_map<int, int>& counts,
                                       const Graph& subgraph) {
        if (current_clique.size() == target_size) {
            for (int v : current_clique) counts[v]++;
            return;
        }

        for (size_t i = 0; i < candidates.size(); ++i) {
            int v = candidates[i];
            bool can_add = true;
            for (int u : current_clique) {
                if (!subgraph.has_edge(u, v)) {
                    can_add = false;
                    break;
                }
            }
            if (can_add) {
                current_clique.push_back(v);
                vector<int> new_candidates;
                for (size_t j = i + 1; j < candidates.size(); ++j) {
                    int u = candidates[j];
                    bool connected = true;
                    for (int w : current_clique) {
                        if (!subgraph.has_edge(w, u)) {
                            connected = false;
                            break;
                        }
                    }
                    if (connected) new_candidates.push_back(u);
                }
                enumerate_cliques_for_density(current_clique, new_candidates, target_size, counts, subgraph);
                current_clique.pop_back();
            }
        }
    }

    struct FlowEdge {
        int to;
        int rev;
        int capacity;
        int flow;
    };

    class FlowNetwork {
    private:
        int n;
        vector<vector<FlowEdge>> adj;
        vector<int> level;
        vector<int> ptr;

    public:
        FlowNetwork(int size) : n(size), adj(size), level(size), ptr(size) {}

        void add_edge(int from, int to, int capacity) {
            adj[from].push_back({to, (int)adj[to].size(), capacity, 0});
            adj[to].push_back({from, (int)adj[from].size() - 1, 0, 0});
        }

        bool bfs(int s, int t) {
            fill(level.begin(), level.end(), -1);
            level[s] = 0;
            queue<int> q;
            q.push(s);

            while (!q.empty()) {
                int v = q.front();
                q.pop();

                for (const auto& edge : adj[v]) {
                    if (level[edge.to] < 0 && edge.flow < edge.capacity) {
                        level[edge.to] = level[v] + 1;
                        q.push(edge.to);
                    }
                }
            }
            return level[t] >= 0;
        }

        int dfs(int v, int t, int flow) {
            if (v == t) return flow;

            for (int& i = ptr[v]; i < adj[v].size(); ++i) {
                auto& edge = adj[v][i];
                if (level[edge.to] == level[v] + 1 && edge.flow < edge.capacity) {
                    int min_flow = min(flow, edge.capacity - edge.flow);
                    int pushed = dfs(edge.to, t, min_flow);
                    if (pushed > 0) {
                        edge.flow += pushed;
                        adj[edge.to][edge.rev].flow -= pushed;
                        return pushed;
                    }
                }
            }
            return 0;
        }

        int max_flow(int s, int t) {
            int total_flow = 0;
            while (bfs(s, t)) {
                fill(ptr.begin(), ptr.end(), 0);
                while (int pushed = dfs(s, t, numeric_limits<int>::max())) {
                    total_flow += pushed;
                }
            }
            return total_flow;
        }

        unordered_set<int> min_cut(int s) {
            unordered_set<int> reachable;
            queue<int> q;
            q.push(s);
            reachable.insert(s);

            while (!q.empty()) {
                int v = q.front();
                q.pop();

                for (const auto& edge : adj[v]) {
                    if (reachable.find(edge.to) == reachable.end() && edge.flow < edge.capacity) {
                        reachable.insert(edge.to);
                        q.push(edge.to);
                    }
                }
            }
            return reachable;
        }
    };

    Graph build_flow_network(const Graph& subgraph, double alpha) {
        unordered_map<int, int> node_to_idx;
        vector<int> idx_to_node;
        int idx = 0;
        
        for (int v : subgraph.get_vertices()) {
            node_to_idx[v] = idx++;
            idx_to_node.push_back(v);
        }

        int n = subgraph.number_of_nodes();
        if (n == 0) return Graph(); // Return empty graph if input is empty
        
        FlowNetwork flow_net(n + 2);
        int source = n;
        int sink = n + 1;

        // Special case for h=2 (edge density)
        if (h == 2) {
            for (int u : subgraph.get_vertices()) {
                int u_idx = node_to_idx[u];
                flow_net.add_edge(source, u_idx, subgraph.neighbors(u).size());
                flow_net.add_edge(u_idx, sink, static_cast<int>(alpha * h));
                
                for (int v : subgraph.neighbors(u)) {
                    if (u < v) { // Avoid double counting
                        int v_idx = node_to_idx[v];
                        flow_net.add_edge(u_idx, v_idx, 1);
                        flow_net.add_edge(v_idx, u_idx, 1);
                    }
                }
            }
        } else {
            // For h > 2, use clique degrees
            for (int v : subgraph.get_vertices()) {
                int v_idx = node_to_idx[v];
                int capacity = clique_degrees.count(v) ? clique_degrees.at(v) : 0;
                flow_net.add_edge(source, v_idx, capacity);
                flow_net.add_edge(v_idx, sink, static_cast<int>(ceil(alpha * h))); // Fix: Use ceil to ensure proper capacity
            }
            
            // Add edges between vertices (for h-cliques)
            for (int u : subgraph.get_vertices()) {
                for (int v : subgraph.neighbors(u)) {
                    if (u < v) {
                        int u_idx = node_to_idx[u];
                        int v_idx = node_to_idx[v];
                        flow_net.add_edge(u_idx, v_idx, n); // Large enough capacity
                        flow_net.add_edge(v_idx, u_idx, n);
                    }
                }
            }
        }

        flow_net.max_flow(source, sink);
        auto reachable = flow_net.min_cut(source);
        
        // Fix: Exclude source vertex from S_nodes
        unordered_set<int> S_nodes;
        for (int i = 0; i < n; ++i) {
            if (reachable.find(i) != reachable.end() && i != source) {
                S_nodes.insert(idx_to_node[i]);
            }
        }
        
        return G.subgraph(S_nodes);
    }

public:
    CoreExact(const Graph& graph, int h_clique) : G(graph), h(h_clique) {
        if (h < 2) {
            throw invalid_argument("h must be at least 2");
        }
        G.optimize_storage();
    }

    Graph run() {
        compute_clique_degrees();
        core_decomposition();
        
        double rho00 = ceil(static_cast<double>(kmax) / h);
        Graph core_graph;
        double max_density = 0.0;
        Graph best_graph; // Track the best subgraph found
        
        unordered_set<int> core_nodes;
        int k00 = static_cast<int>(rho00);
        for (const auto& pair : core_numbers) {
            if (pair.second >= k00) {
                core_nodes.insert(pair.first);
            }
        }
        core_graph = G.subgraph(core_nodes);
        
        if (core_graph.number_of_nodes() == 0) {
            // If no nodes satisfy k >= rho00, try with k=1
            for (const auto& pair : core_numbers) {
                if (pair.second >= 1) {
                    core_nodes.insert(pair.first);
                }
            }
            core_graph = G.subgraph(core_nodes);
        }
        
        auto components = connected_components(core_graph);
        
        for (const auto& component : components) {
            Graph comp_graph = G.subgraph(component);
            double density = compute_density(comp_graph);
            if (density > max_density) {
                max_density = density;
                best_graph = comp_graph;
            }
        }
        
        k00 = max(1, static_cast<int>(ceil(max_density)));
        
        core_nodes.clear();
        for (const auto& pair : core_numbers) {
            if (pair.second >= k00) {
                core_nodes.insert(pair.first);
            }
        }
        core_graph = G.subgraph(core_nodes);
        components = connected_components(core_graph);
        
        Graph D = best_graph; // Start with the best graph found so far
        double l = max_density;
        double u = kmax;
        
        for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
            const auto& component = components[comp_idx];
            
            Graph C = G.subgraph(component);
            int n_C = C.number_of_nodes();
            
            if (n_C <= 1) continue; // Skip trivial components
            
            // Skip if component's max core number is less than current lower bound
            int max_core = 0;
            for (int v : component) {
                if (core_numbers.count(v) && core_numbers[v] > max_core) {
                    max_core = core_numbers[v];
                }
            }
            
            if (max_core < l) {
                continue;
            }
            
            // For h=2, we can use a simpler approach for density calculation
            if (h == 2 && n_C > 0) {
                double density = static_cast<double>(C.number_of_edges()) / n_C;
                if (density > max_density) {
                    max_density = density;
                    D = C;
                }
            }
            
            // Binary search for optimal alpha
            double epsilon = 1.0 / (n_C * (n_C - 1));
            
            int iteration = 0;
            int max_iterations = 100; // Safety limit to prevent infinite loops
            double prev_density = -1.0; // To track previous density for convergence check
            
            // Fix: Set local lower and upper bounds for the binary search
            double local_l = l;
            double local_u = max_core;
            
            while (local_u - local_l >= epsilon && iteration < max_iterations) {
                iteration++;
                double alpha = (local_l + local_u) / 2;
                
                Graph S_graph = build_flow_network(C, alpha);
                
                if (S_graph.number_of_nodes() == 0) {
                    local_u = alpha;
                } else {
                    double density = compute_density(S_graph);
                    
                    // Check if we're making progress
                    if (fabs(density - prev_density) < epsilon) {
                        break;
                    }
                    prev_density = density;
                    
                    if (density > local_l) {
                        local_l = density;
                        
                        // Filter nodes with low core numbers
                        int min_core = numeric_limits<int>::max();
                        for (int v : S_graph.get_vertices()) {
                            if (core_numbers.count(v) && core_numbers[v] < min_core) {
                                min_core = core_numbers[v];
                            }
                        }
                        
                        if (local_l > min_core) {
                            unordered_set<int> new_core_nodes;
                            int ceil_l = static_cast<int>(ceil(local_l));
                            for (int v : C.get_vertices()) {
                                if (core_numbers.count(v) && core_numbers[v] >= ceil_l) {
                                    new_core_nodes.insert(v);
                                }
                            }
                            C = G.subgraph(new_core_nodes);
                            n_C = C.number_of_nodes();
                        }
                    } else {
                        local_u = alpha;
                    }
                    
                    // Update global best density and subgraph
                    if (density > max_density) {
                        D = S_graph;
                        max_density = density;
                        l = density; // Update global lower bound too
                    }
                }
            }
            
            // Store final bounds for the last component processed
            final_lower_bound = local_l;
            final_upper_bound = local_u;
            
            // Check if the whole component is better
            double C_density = compute_density(C);
            if (C_density > max_density) {
                D = C;
                max_density = C_density;
            }
        }
        
        return D;
    }

    void print_results(const Graph& densest, ostream& out = cout) {
        int edge_count = 0;
        for (int u : densest.get_vertices()) {
            for (int v : densest.neighbors(u)) {
                if (u < v) {
                    edge_count++;
                }
            }
        }
        
        out << "Total nodes: " << densest.number_of_nodes() << endl;
        out << "Total edges: " << edge_count << endl;
        out << "Density: " << compute_density(densest) << endl;
        out << "Final lower bound: " << final_lower_bound << endl;
        out << "Final upper bound: " << final_upper_bound << endl;
    }
};

Graph read_graph_from_file(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    Graph G;
    int u, v;
    int line_count = 0;
    int edge_count = 0;
    string line;
    
    try {
        while (getline(file, line)) {
            line_count++;
            // Skip empty lines or comments
            if (line.empty() || line[0] == '#') continue;
            
            istringstream iss(line);
            if (!(iss >> u >> v)) {
                cerr << "Warning: Invalid format at line " << line_count << ": " << line << endl;
                continue;
            }
            G.add_edge(u, v);
            edge_count++;
        }
    } catch (const exception& e) {
        cerr << "Error reading line " << line_count << ": " << e.what() << endl;
        throw;
    }
    
    return G;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <h_value> [-o output_file]\n";
        return 1;
    }

    string input_file = argv[1];
    int h;
    try {
        h = stoi(argv[2]);
    } catch (const invalid_argument& e) {
        cerr << "Error: Invalid h value '" << argv[2] << "'\n";
        return 1;
    }

    string output_file;
    bool output_to_file = false;
    if (argc >= 5 && string(argv[3]) == "-o") {
        output_file = argv[4];
        output_to_file = true;
    }

    try {
        Graph G = read_graph_from_file(input_file);
        CoreExact core_exact(G, h);
        Graph densest = core_exact.run();

        if (output_to_file) {
            ofstream out(output_file);
            if (!out.is_open()) {
                throw runtime_error("Could not open output file: " + output_file);
            }
            core_exact.print_results(densest, out);
        } else {
            core_exact.print_results(densest);
        }
    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }

    return 0;
}