#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mesh.hpp"

// ---------------------------------------------------------------------------
// Union-Find (for β₀)
// ---------------------------------------------------------------------------

struct UnionFind {
    std::vector<int> parent, rnk;
    int components;

    explicit UnionFind(int n) : parent(n), rnk(n, 0), components(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rnk[a] < rnk[b]) std::swap(a, b);
        parent[b] = a;
        if (rnk[a] == rnk[b]) rnk[a]++;
        components--;
        return true;
    }
};

// ---------------------------------------------------------------------------
// GF(2) rank of ∂₂ via sparse column reduction
//
// Each column is a sorted list of edge indices bounding one face.
// Standard persistence/Smith-normal-form reduction over GF(2):
//   for each face-column, reduce against stored pivot columns keyed by their
//   lowest edge index until the column vanishes (face is a boundary; rank
//   doesn't increase) or reveals a new pivot (rank increases).
// ---------------------------------------------------------------------------

static std::vector<int> sym_diff(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                  std::back_inserter(out));
    return out;
}

static int rank_boundary2(std::vector<std::vector<int>> cols) {
    // pivot[lowest_edge_in_col] = reduced column
    std::unordered_map<int, std::vector<int>> pivot;
    pivot.reserve(cols.size());
    int rnk = 0;
    for (auto& col : cols) {
        while (!col.empty()) {
            int low = col[0];
            auto it = pivot.find(low);
            if (it == pivot.end()) {
                pivot.emplace(low, col);
                rnk++;
                break;
            }
            col = sym_diff(col, it->second);
        }
    }
    return rnk;
}

// ---------------------------------------------------------------------------
// Graph file loader (plain edge list or DIMACS format)
//
// Supports:
//   "u v"           — plain 0-based or 1-based pairs
//   "e u v"         — DIMACS edge line (1-based)
//   "p graph V E"   — DIMACS header (sets 1-based mode)
//   lines starting with 'c' are ignored (comments)
// ---------------------------------------------------------------------------

static std::pair<int, std::vector<std::array<int,2>>>
load_graph(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    int V = 0, max_id = -1;
    bool one_based = false;
    std::vector<std::array<int,2>> edges;
    std::string line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::istringstream ss(line);
        if (line[0] == 'p') {
            std::string tok;
            int ne;
            ss >> tok >> tok >> V >> ne;
            one_based = true;
            edges.reserve(ne);
            continue;
        }
        std::string tok;
        ss >> tok;
        int u, v;
        if (tok == "e" || tok == "a") {
            ss >> u >> v;
        } else {
            u = std::stoi(tok);
            if (!(ss >> v)) continue;
        }
        if (one_based) { u--; v--; }
        max_id = std::max({max_id, u, v});
        edges.push_back({u, v});
    }

    if (V == 0) V = max_id + 1;
    return {V, edges};
}

// ---------------------------------------------------------------------------
// Classify mesh topology (heuristic, printed as a hint)
// ---------------------------------------------------------------------------

static void print_topology_hint(int beta0, int beta1, int beta2, long long chi) {
    // For a closed orientable 2-manifold each component contributes β₂ = 1.
    // If β₂ == β₀ the mesh is likely a closed surface (no boundary).
    if (beta2 == beta0 && beta0 > 0) {
        // χ = 2β₀ − β₁  →  genus_total = β₁/2
        if (beta1 % 2 == 0) {
            int genus_total = beta1 / 2;
            std::cout << "  → Closed orientable surface: total genus = "
                      << genus_total;
            if (beta0 == 1)
                std::cout << "  (sphere if g=0, torus if g=1, …)";
            std::cout << "\n";
        }
    } else if (beta2 == 0 && beta0 > 0) {
        std::cout << "  → Surface has boundary (or non-manifold / open mesh)\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage(const char* argv0) {
    std::cerr <<
        "Usage: " << argv0 << " <input.(obj|ply|gii|graph)>\n"
        "\n"
        "Computes the Betti numbers of a mesh or graph via GF(2) simplicial\n"
        "homology.  The input is treated as a 2-complex (or 1-complex for\n"
        "graphs).\n"
        "\n"
        "  β₀  connected components\n"
        "  β₁  independent 1-cycles (loops / handles)\n"
        "  β₂  independent 2-cycles (enclosed voids)\n"
        "\n"
        "For .graph files: plain edge list, one 'u v' per line (0-based).\n"
        "DIMACS format ('p graph V E' / 'e u v' lines) is also accepted.\n"
        "\n"
        "Algorithm:\n"
        "  1. Build edge list and face list from input.\n"
        "  2. β₀ via union-find; rank(∂₁) = V − β₀.\n"
        "  3. rank(∂₂) via sparse GF(2) column reduction of the boundary\n"
        "     matrix (each face column = its 3 edge indices).\n"
        "  4. β₁ = (E − rank(∂₁)) − rank(∂₂)\n"
        "     β₂ = F − rank(∂₂)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string path = argv[1];
    std::string ext;
    {
        auto dot = path.rfind('.');
        if (dot != std::string::npos) {
            ext = path.substr(dot + 1);
            for (auto& c : ext) c = static_cast<char>(std::tolower(c));
        }
    }

    using Clock = std::chrono::steady_clock;
    using Sec   = std::chrono::duration<double>;
    auto t0 = Clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
    };

    try {
        int V = 0;
        std::vector<std::array<int,2>> raw_edges;
        std::vector<std::array<int,3>> faces;

        // ---- Load input -------------------------------------------------------
        if (ext == "obj" || ext == "ply" || ext == "gii") {
            std::cerr << "Loading mesh: " << path << " ...\n";
            Mesh m = load_mesh(path);
            V     = static_cast<int>(m.vertices.size());
            faces = m.triangles;
            raw_edges.reserve(3 * faces.size());
            for (auto& t : faces)
                for (int i = 0; i < 3; i++) {
                    int a = t[i], b = t[(i+1)%3];
                    if (a > b) std::swap(a, b);
                    raw_edges.push_back({a, b});
                }
            std::cerr << "  " << V << " vertices, " << faces.size()
                      << " triangles  [" << elapsed() << " s]\n";
        } else {
            std::cerr << "Loading graph: " << path << " ...\n";
            auto [nv, ge] = load_graph(path);
            V = nv;
            raw_edges = ge;
            for (auto& e : raw_edges)
                if (e[0] > e[1]) std::swap(e[0], e[1]);
            std::cerr << "  " << V << " vertices, " << raw_edges.size()
                      << " edges  [" << elapsed() << " s]\n";
        }

        // ---- Build canonical edge index --------------------------------------
        std::sort(raw_edges.begin(), raw_edges.end());
        raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()),
                        raw_edges.end());
        int E = static_cast<int>(raw_edges.size());
        int F = static_cast<int>(faces.size());

        // Map edge pair → sequential index
        std::map<std::array<int,2>, int> edge_idx;
        for (int i = 0; i < E; i++) edge_idx[raw_edges[i]] = i;

        // ---- β₀ via union-find -----------------------------------------------
        std::cerr << "Computing β₀ ...\n";
        UnionFind uf(V);
        for (auto& e : raw_edges) uf.unite(e[0], e[1]);
        int beta0    = uf.components;
        int rank_d1  = V - beta0;   // rank of ∂₁ (from spanning-forest argument)

        std::cerr << "  β₀ = " << beta0 << "  [" << elapsed() << " s]\n";

        // ---- rank(∂₂) via GF(2) column reduction ----------------------------
        int rank_d2 = 0;
        if (F > 0) {
            std::cerr << "Computing rank(∂₂) over GF(2) ...\n";
            std::vector<std::vector<int>> cols;
            cols.reserve(F);
            for (auto& t : faces) {
                std::vector<int> boundary;
                boundary.reserve(3);
                for (int i = 0; i < 3; i++) {
                    int a = t[i], b = t[(i+1)%3];
                    if (a > b) std::swap(a, b);
                    boundary.push_back(edge_idx.at({a, b}));
                }
                std::sort(boundary.begin(), boundary.end());
                cols.push_back(std::move(boundary));
            }
            rank_d2 = rank_boundary2(std::move(cols));
            std::cerr << "  rank(∂₂) = " << rank_d2
                      << "  [" << elapsed() << " s]\n";
        }

        // ---- Betti numbers ---------------------------------------------------
        int beta1 = (E - rank_d1) - rank_d2;
        int beta2 = F - rank_d2;
        long long chi = static_cast<long long>(V) - E + F;

        // ---- Report ----------------------------------------------------------
        std::cout << "\n";
        std::cout << "Input          : " << path << "\n";
        std::cout << "Simplex counts : V=" << V
                  << "  E=" << E << "  F=" << F << "\n";
        std::cout << "Euler χ        : " << chi
                  << "  (= β₀ − β₁ + β₂ = "
                  << beta0 << " − " << beta1 << " + " << beta2 << " = "
                  << (beta0 - beta1 + beta2) << ")\n";
        std::cout << "\nBetti numbers:\n";
        std::cout << "  β₀ = " << beta0 << "   connected components\n";
        std::cout << "  β₁ = " << beta1 << "   independent 1-cycles (loops / handles)\n";
        std::cout << "  β₂ = " << beta2 << "   independent 2-cycles (enclosed voids)\n";
        std::cout << "\n";
        print_topology_hint(beta0, beta1, beta2, chi);

        std::cerr << "\nTotal time: " << elapsed() << " s\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
