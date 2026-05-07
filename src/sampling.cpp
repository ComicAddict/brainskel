#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Spatial grid for Poisson-disk rejection
// ---------------------------------------------------------------------------

struct Grid3D {
    Vec3 origin;
    double cell_size;
    int nx, ny, nz;
    // cell -> list of sample indices
    std::unordered_map<int, std::vector<int>> cells;

    Grid3D(const Vec3& mn, const Vec3& mx, double r)
        : origin(mn), cell_size(r) {
        nx = static_cast<int>((mx.x - mn.x) / r) + 2;
        ny = static_cast<int>((mx.y - mn.y) / r) + 2;
        nz = static_cast<int>((mx.z - mn.z) / r) + 2;
    }

    std::array<int, 3> cell_of(const Vec3& p) const {
        return {
            static_cast<int>((p.x - origin.x) / cell_size),
            static_cast<int>((p.y - origin.y) / cell_size),
            static_cast<int>((p.z - origin.z) / cell_size)
        };
    }

    int flat(int ix, int iy, int iz) const {
        return ix * ny * nz + iy * nz + iz;
    }

    void insert(const Vec3& p, int sample_idx) {
        auto [cx, cy, cz] = cell_of(p);
        cells[flat(cx, cy, cz)].push_back(sample_idx);
    }

    // Returns true if any existing sample is within min_dist of p.
    bool has_conflict(const Vec3& p, const std::vector<Vec3>& positions,
                      double min_dist) const {
        auto [cx, cy, cz] = cell_of(p);
        double r2 = min_dist * min_dist;

        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
            int ix = cx + dx, iy = cy + dy, iz = cz + dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz)
                continue;
            auto it = cells.find(flat(ix, iy, iz));
            if (it == cells.end()) continue;
            for (int idx : it->second) {
                Vec3 d = p - positions[idx];
                if (d.dot(d) < r2) return true;
            }
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Poisson-disk sampling
// ---------------------------------------------------------------------------

SampledPoints poisson_disk_sample(const Mesh& mesh, double min_radius,
                                   unsigned int seed) {
    if (mesh.triangles.empty())
        throw std::runtime_error("Mesh has no triangles.");

    // Build cumulative triangle-area distribution for uniform surface sampling.
    int ntri = static_cast<int>(mesh.triangles.size());
    std::vector<double> cum_area(ntri + 1, 0.0);
    for (int i = 0; i < ntri; i++) {
        const auto& t = mesh.triangles[i];
        Vec3 e1 = mesh.vertices[t[1]] - mesh.vertices[t[0]];
        Vec3 e2 = mesh.vertices[t[2]] - mesh.vertices[t[0]];
        cum_area[i + 1] = cum_area[i] + 0.5 * e1.cross(e2).norm();
    }
    double total_area = cum_area[ntri];
    if (total_area < 1e-15)
        throw std::runtime_error("Mesh has zero surface area.");

    // Bounding box for grid.
    Vec3 bmin = mesh.vertices[0], bmax = mesh.vertices[0];
    for (const auto& v : mesh.vertices) {
        bmin.x = std::min(bmin.x, v.x); bmax.x = std::max(bmax.x, v.x);
        bmin.y = std::min(bmin.y, v.y); bmax.y = std::max(bmax.y, v.y);
        bmin.z = std::min(bmin.z, v.z); bmax.z = std::max(bmax.z, v.z);
    }

    Grid3D grid(bmin, bmax, min_radius);

    // Number of random candidates to throw.  Oversample by ~10x so that
    // the Poisson disk fills up well even with a pure dart-throwing approach.
    double area_per_disk = M_PI * min_radius * min_radius / 4.0;
    int n_expected = static_cast<int>(total_area / area_per_disk);
    int n_candidates = std::max(n_expected * 10, 50000);

    SampledPoints result;
    result.positions.reserve(n_expected);
    result.normals.reserve(n_expected);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int iter = 0; iter < n_candidates; iter++) {
        // Pick a random triangle weighted by area.
        double ra = uni(rng) * total_area;
        int tri = static_cast<int>(
            std::lower_bound(cum_area.begin(), cum_area.end(), ra) -
            cum_area.begin()) - 1;
        tri = std::clamp(tri, 0, ntri - 1);

        const auto& t = mesh.triangles[tri];
        const Vec3& va = mesh.vertices[t[0]];
        const Vec3& vb = mesh.vertices[t[1]];
        const Vec3& vc = mesh.vertices[t[2]];

        // Uniform random point in triangle via barycentric coords.
        double u = uni(rng), v = uni(rng);
        if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
        double w = 1.0 - u - v;

        Vec3 pos = va * w + vb * u + vc * v;
        Vec3 nrm = (mesh.normals[t[0]] * w +
                    mesh.normals[t[1]] * u +
                    mesh.normals[t[2]] * v).normalized();

        if (grid.has_conflict(pos, result.positions, min_radius))
            continue;

        grid.insert(pos, static_cast<int>(result.positions.size()));
        result.positions.push_back(pos);
        result.normals.push_back(nrm);
    }

    return result;
}

// ---------------------------------------------------------------------------
// I/O  (PLY ASCII with x y z nx ny nz — importable in Blender as point cloud)
// ---------------------------------------------------------------------------

void save_points(const std::string& path, const SampledPoints& pts) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << "ply\n"
      << "format ascii 1.0\n"
      << "element vertex " << pts.positions.size() << "\n"
      << "property double x\n"
      << "property double y\n"
      << "property double z\n"
      << "property double nx\n"
      << "property double ny\n"
      << "property double nz\n"
      << "end_header\n";

    f.precision(10);
    for (size_t i = 0; i < pts.positions.size(); i++) {
        const auto& p = pts.positions[i];
        const auto& n = pts.normals[i];
        f << p.x << ' ' << p.y << ' ' << p.z << ' '
          << n.x << ' ' << n.y << ' ' << n.z << '\n';
    }
}

SampledPoints load_points(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(f, line);

    // PLY file
    if (line == "ply") {
        int n_verts = 0;
        // Minimal header parse: find "element vertex N" and "end_header".
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "element") {
                std::string name; ss >> name >> n_verts;
            } else if (tok == "end_header") {
                break;
            }
        }
        SampledPoints pts;
        pts.positions.reserve(n_verts);
        pts.normals.reserve(n_verts);
        for (int i = 0; i < n_verts && std::getline(f, line); i++) {
            std::istringstream ss(line);
            Vec3 p, n;
            if (ss >> p.x >> p.y >> p.z >> n.x >> n.y >> n.z) {
                pts.positions.push_back(p);
                pts.normals.push_back(n);
            }
        }
        return pts;
    }

    // Legacy plain-text fallback: first line was already read, re-process it.
    SampledPoints pts;
    auto parse_line = [&](const std::string& ln) {
        if (ln.empty() || ln[0] == '#') return;
        std::istringstream ss(ln);
        Vec3 p, n;
        if (ss >> p.x >> p.y >> p.z >> n.x >> n.y >> n.z) {
            pts.positions.push_back(p);
            pts.normals.push_back(n);
        }
    };
    parse_line(line);
    while (std::getline(f, line)) parse_line(line);
    return pts;
}
