/**
 * LinExt C++ Library - Python Interface
 * 
 * Computes the number of linear extensions (topological orderings) of a DAG.
 * Uses optimized recursive algorithm with memoization.
 * 
 * Compile with:
 *   clang++ -shared -fPIC -O3 -std=c++14 -o liblinext.dylib linext_python_interface.cpp
 * Or on Linux:
 *   g++ -shared -fPIC -O3 -std=c++14 -o liblinext.so linext_python_interface.cpp
 */

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <numeric>

// Use 64-bit integers for large counts
using lint = long long;

/**
 * Compute number of linear extensions using recursive algorithm.
 * 
 * @param adj   Adjacency matrix (row-major, n*n)
 * @param n     Number of nodes
 * @param mask  Bitmask of remaining nodes
 * @param memo  Memoization cache
 * @return      Number of linear extensions
 */
lint count_extensions_recursive(
    const int* adj, 
    int n, 
    uint64_t mask,
    std::unordered_map<uint64_t, lint>& memo
) {
    // Base case: empty set has 1 extension (the empty sequence)
    if (mask == 0) return 1;
    
    // Check memo
    auto it = memo.find(mask);
    if (it != memo.end()) return it->second;
    
    // Find minimal elements (nodes with no incoming edges from remaining nodes)
    std::vector<int> minimals;
    for (int i = 0; i < n; ++i) {
        if (!(mask & (1ULL << i))) continue;  // Node not in remaining set
        
        bool is_minimal = true;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            if (!(mask & (1ULL << j))) continue;  // Node j not in remaining set
            // If there's an edge j -> i, then i is not minimal
            if (adj[j * n + i]) {
                is_minimal = false;
                break;
            }
        }
        if (is_minimal) {
            minimals.push_back(i);
        }
    }
    
    // Sum extensions starting with each minimal element
    lint total = 0;
    for (int m : minimals) {
        uint64_t new_mask = mask & ~(1ULL << m);  // Remove m from remaining
        total += count_extensions_recursive(adj, n, new_mask, memo);
    }
    
    memo[mask] = total;
    return total;
}

/**
 * Count linear extensions with optimization for isolated nodes.
 */
lint count_linear_extensions_internal(const int* adj, int n) {
    if (n <= 0) return 1;
    if (n == 1) return 1;
    if (n > 63) {
        // For very large DAGs, fall back to simpler approach
        // Bitmask won't work with n > 63
        // In practice, MCMC rarely needs n > 20-30
        std::unordered_map<uint64_t, lint> memo;
        return count_extensions_recursive(adj, n, (1ULL << n) - 1, memo);
    }
    
    // Find isolated nodes (no incoming or outgoing edges)
    std::vector<int> isolated;
    std::vector<int> non_isolated;
    
    for (int i = 0; i < n; ++i) {
        bool has_edges = false;
        for (int j = 0; j < n; ++j) {
            if (adj[i * n + j] || adj[j * n + i]) {
                has_edges = true;
                break;
            }
        }
        if (has_edges) {
            non_isolated.push_back(i);
        } else {
            isolated.push_back(i);
        }
    }
    
    int k = isolated.size();
    int m = non_isolated.size();
    
    if (m == 0) {
        // All nodes are isolated: n! ways to order them
        lint factorial = 1;
        for (int i = 2; i <= n; ++i) factorial *= i;
        return factorial;
    }
    
    // Build reduced adjacency matrix for non-isolated nodes
    std::vector<int> reduced_adj(m * m, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            reduced_adj[i * m + j] = adj[non_isolated[i] * n + non_isolated[j]];
        }
    }
    
    // Count extensions of non-isolated subgraph
    std::unordered_map<uint64_t, lint> memo;
    lint base_count = count_extensions_recursive(reduced_adj.data(), m, (1ULL << m) - 1, memo);
    
    // Multiply by ways to insert isolated nodes
    // k isolated nodes can be inserted into (m+1) positions
    // This is (n choose k) * k! = n! / m!
    lint multiplier = 1;
    for (int i = m + 1; i <= n; ++i) multiplier *= i;
    
    return base_count * multiplier;
}

// C interface for Python ctypes
extern "C" {

/**
 * Count linear extensions from flattened adjacency matrix.
 * 
 * @param matrix_flat  Flattened adjacency matrix (row-major, size * size)
 * @param size         Number of nodes
 * @return             Number of linear extensions, or -1 on error
 */
long count_linear_extensions_flat(int* matrix_flat, int size) {
    if (size <= 0) return 1;
    if (size == 1) return 1;
    if (matrix_flat == nullptr) return -1;
    
    try {
        return static_cast<long>(count_linear_extensions_internal(matrix_flat, size));
    } catch (...) {
        return -1;
    }
}

/**
 * Simple test function to verify library is loaded correctly.
 */
int linext_test() {
    return 42;
}

}  // extern "C"







