#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <queue>
#include <limits>
#include <map>
#include <cmath>
#include <functional>

class Graph {
public:
    int n;
    int m;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<bool>> adjMatrix;
    
    Graph(int n) : n(n), m(0), adj(n), adjMatrix(n, std::vector<bool>(n, false)) {}
    
    void addEdge(int u, int v) {
        if (u == v || adjMatrix[u][v]) return;
        
        adj[u].push_back(v);
        adj[v].push_back(u);
        adjMatrix[u][v] = adjMatrix[v][u] = true;
        m++;
    }
    
    bool isConnected(int u, int v) const {
        if (u >= n || v >= n) return false;
        return adjMatrix[u][v];
    }
    
    int cliqueDegree(int v, const std::vector<std::vector<int>>& hMinusOneCliques) const {
        int count = 0;
        for (const auto& clique : hMinusOneCliques) {
            bool canFormClique = true;
            for (int u : clique) {
                if (!isConnected(v, u)) {
                    canFormClique = false;
                    break;
                }
            }
            if (canFormClique) count++;
        }
        return count;
    }
    
    int maxCliqueDegree(const std::vector<std::vector<int>>& hMinusOneCliques) const {
        int maxDeg = 0;
        for (int v = 0; v < n; v++) {
            maxDeg = std::max(maxDeg, cliqueDegree(v, hMinusOneCliques));
        }
        return maxDeg;
    }
    
    std::vector<std::vector<int>> findHMinusOneCliques(int h) const {
        std::vector<std::vector<int>> result;
        
        if (h <= 1) return result;
        
        if (h == 2) {
            for (int v = 0; v < n; v++) {
                result.push_back({v});
            }
            return result;
        }
        
        if (h == 3) {
            for (int u = 0; u < n; u++) {
                for (int v : adj[u]) {
                    if (u < v) {
                        result.push_back({u, v});
                    }
                }
            }
            return result;
        }
        
        std::vector<int> R, P, X;
        P.reserve(n);
        for (int i = 0; i < n; i++) {
            P.push_back(i);
        }
        
        std::function<void(std::vector<int>&, std::vector<int>&, std::vector<int>&)> bronKerbosch = 
            [&](std::vector<int>& R, std::vector<int>& P, std::vector<int>& X) {
                if (P.empty() && X.empty()) {
                    if (R.size() == h - 1) {
                        result.push_back(R);
                    }
                    return;
                }
                
                std::vector<int> P_copy = P;
                for (int v : P_copy) {
                    R.push_back(v);
                    
                    std::vector<int> P_new;
                    for (int u : P) {
                        if (isConnected(u, v)) {
                            P_new.push_back(u);
                        }
                    }
                    
                    std::vector<int> X_new;
                    for (int u : X) {
                        if (isConnected(u, v)) {
                            X_new.push_back(u);
                        }
                    }
                    
                    bronKerbosch(R, P_new, X_new);
                    
                    R.pop_back();
                    
                    P.erase(std::remove(P.begin(), P.end(), v), P.end());
                    X.push_back(v);
                }
            };
        
        bronKerbosch(R, P, X);
        return result;
    }
    
    bool formsHClique(int v, const std::vector<int>& hMinusOneClique) const {
        for (int u : hMinusOneClique) {
            if (!isConnected(v, u)) return false;
        }
        return true;
    }
    
    Graph induceSubgraph(const std::vector<int>& vertices) const {
        if (vertices.empty()) return Graph(0);
        
        std::map<int, int> vertexMap;
        for (size_t i = 0; i < vertices.size(); i++) {
            vertexMap[vertices[i]] = i;
        }
        
        Graph subgraph(vertices.size());
        
        for (size_t i = 0; i < vertices.size(); i++) {
            int v = vertices[i];
            for (int u : adj[v]) {
                auto it = vertexMap.find(u);
                if (it != vertexMap.end() && it->second > i) {
                    subgraph.addEdge(i, it->second);
                }
            }
        }
        
        return subgraph;
    }
};

class FlowNetwork {
public:
    int n;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<long long>> capacity;
    static const long long INF = 1e18;
    
    FlowNetwork(int n) : n(n), adj(n), capacity(n, std::vector<long long>(n, 0)) {}
    
    void addEdge(int u, int v, long long cap) {
        adj[u].push_back(v);
        adj[v].push_back(u);
        capacity[u][v] = cap;
    }
    
    std::pair<long long, std::vector<int>> minCut(int s, int t) {
        std::vector<std::vector<long long>> residual = capacity;
        long long max_flow = 0;
        
        while (true) {
            std::vector<int> parent(n, -1);
            std::queue<int> q;
            q.push(s);
            parent[s] = -2;
            
            while (!q.empty() && parent[t] == -1) {
                int u = q.front();
                q.pop();
                
                for (int v : adj[u]) {
                    if (parent[v] == -1 && residual[u][v] > 0) {
                        parent[v] = u;
                        q.push(v);
                    }
                }
            }
            
            if (parent[t] == -1) break;
            
            long long path_flow = INF;
            for (int v = t; v != s; v = parent[v]) {
                int u = parent[v];
                path_flow = std::min(path_flow, residual[u][v]);
            }
            
            for (int v = t; v != s; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= path_flow;
                residual[v][u] += path_flow;
            }
            
            max_flow += path_flow;
        }
        
        std::vector<int> S;
        std::vector<bool> visited(n, false);
        std::queue<int> q;
        q.push(s);
        visited[s] = true;
        S.push_back(s);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v : adj[u]) {
                if (!visited[v] && residual[u][v] > 0) {
                    visited[v] = true;
                    q.push(v);
                    S.push_back(v);
                }
            }
        }
        
        return {max_flow, S};
    }
};

// Determine if the graph is 0-indexed or 1-indexed based on input
Graph readGraphFile(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("File not found: " + fileName);
    }
    
    std::string line;
    int maxVertex = -1;
    int minVertex = std::numeric_limits<int>::max();
    std::vector<std::pair<int, int>> edges;
    
    // Check file extension
    std::string ext = fileName.substr(fileName.find_last_of('.') + 1);
    bool isMTX = (ext == "mtx" || ext == "MTX");
    
    // Skip comments in mtx format
    if (isMTX) {
        while (std::getline(file, line)) {
            if (line.empty() || line[0] != '%') break;
        }
    } else {
        std::getline(file, line);  // For txt, just read the first line
    }
    
    int rows = 0, cols = 0, entries = 0;
    
    // Parse first non-comment line for mtx
    if (isMTX) {
        std::istringstream iss(line);
        iss >> rows >> cols >> entries;
    }
    
    // Read all edges and determine min/max vertex ID
    int u, v;
    std::string currentLine = isMTX ? line : "";
    
    while (true) {
        if (!isMTX || edges.empty()) {  // For non-MTX or first edge in MTX
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                if (!(iss >> u >> v)) continue;  // Skip invalid lines
            } else {
                break;  // End of file
            }
        } else {
            if (file >> u >> v) {  // Continue with MTX format
                // Process normally
            } else {
                break;  // End of file or error
            }
        }
        
        edges.push_back({u, v});
        maxVertex = std::max(maxVertex, std::max(u, v));
        minVertex = std::min(minVertex, std::min(u, v));
    }
    
    // Determine if 0-indexed or 1-indexed
    bool isZeroIndexed = (minVertex == 0);
    int vertexCount = maxVertex + (isZeroIndexed ? 1 : 0);
    
    std::cout << "Detected " << (isZeroIndexed ? "0-indexed" : "1-indexed") << " graph with " 
              << vertexCount << " vertices" << std::endl;
    
    Graph graph(vertexCount);
    
    // Add edges with appropriate indexing
    for (const auto& edge : edges) {
        int adjustedU = isZeroIndexed ? edge.first : edge.first - 1;
        int adjustedV = isZeroIndexed ? edge.second : edge.second - 1;
        graph.addEdge(adjustedU, adjustedV);
    }
    
    return graph;
}

Graph exactDensestSubgraph(const Graph& G, int h) {
    int n = G.n;
    
    double l = 0;
    std::vector<int> D;
    
    std::vector<std::vector<int>> Lambda = G.findHMinusOneCliques(h);
    
    double u = G.maxCliqueDegree(Lambda);
    
    int iteration = 0;
    while (u - l >= 1.0 / (n * (n - 1))) {
        iteration++;
        double alpha = (l + u) / 2;
        
        int vF = n + Lambda.size() + 2;
        int s = n + Lambda.size();
        int t = s + 1;
        
        FlowNetwork network(vF);
        
        for (int v = 0; v < n; v++) {
            int degree = G.cliqueDegree(v, Lambda);
            network.addEdge(s, v, degree);
            network.addEdge(v, t, static_cast<long long>(alpha * h));
        }
        
        for (size_t i = 0; i < Lambda.size(); i++) {
            int cliqueNode = n + i;
            for (int v : Lambda[i]) {
                network.addEdge(cliqueNode, v, FlowNetwork::INF);
            }
        }
        
        for (size_t i = 0; i < Lambda.size(); i++) {
            int cliqueNode = n + i;
            for (int v = 0; v < n; v++) {
                if (G.formsHClique(v, Lambda[i])) {
                    network.addEdge(v, cliqueNode, 1);
                }
            }
        }
        
        auto [maxFlow, S] = network.minCut(s, t);
        
        if (S.size() == 1 && S[0] == s) {
            u = alpha;
        } else {
            l = alpha;
            
            D.clear();
            for (int v : S) {
                if (v != s && v < n) {
                    D.push_back(v);
                }
            }
        }
    }
    
    std::cout << "Final bounds: l = " << l << ", u = " << u << std::endl;
    std::cout << "Densest subgraph size: " << D.size() << " vertices" << std::endl;
    
    return G.induceSubgraph(D);
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <input_file> <h_value>" << std::endl;
            return 1;
        }
        
        std::string fileName = argv[1];
        int h = std::stoi(argv[2]);
        
        Graph G = readGraphFile(fileName);
        
        std::cout << "Graph has " << G.n << " vertices and " << G.m << " edges" << std::endl;
        
        Graph densestSubgraph = exactDensestSubgraph(G, h);
        
        std::cout << "Densest subgraph has " << densestSubgraph.n << " vertices and " 
                << densestSubgraph.m << " edges" << std::endl;
                
        auto hMinusOneCliques = densestSubgraph.findHMinusOneCliques(h);
        int numCliques = 0;
        
        for (int v = 0; v < densestSubgraph.n; v++) {
            numCliques += densestSubgraph.cliqueDegree(v, hMinusOneCliques);
        }
        numCliques /= h;
        
        double density = static_cast<double>(numCliques) / densestSubgraph.n;
        std::cout << "h-clique-density: " << density << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}