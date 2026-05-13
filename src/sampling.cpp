#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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

void save_points(const std::string& path, const SampledPoints& pts,
                 bool binary, double sampling_radius) {
    auto mode = binary ? (std::ios::binary | std::ios::out) : std::ios::out;
    std::ofstream f(path, mode);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << "ply\n"
      << (binary ? "format binary_little_endian 1.0\n" : "format ascii 1.0\n");
    if (sampling_radius > 0.0) {
        f.precision(17);
        f << "comment sampling_radius " << sampling_radius << "\n";
    }
    f << "element vertex " << pts.positions.size() << "\n"
      << "property double x\n"
      << "property double y\n"
      << "property double z\n"
      << "property double nx\n"
      << "property double ny\n"
      << "property double nz\n"
      << "end_header\n";

    if (binary) {
        for (size_t i = 0; i < pts.positions.size(); i++) {
            double row[6] = {pts.positions[i].x, pts.positions[i].y, pts.positions[i].z,
                             pts.normals[i].x,   pts.normals[i].y,   pts.normals[i].z};
            f.write(reinterpret_cast<const char*>(row), 48);
        }
    } else {
        f.precision(10);
        for (size_t i = 0; i < pts.positions.size(); i++) {
            const auto& p = pts.positions[i];
            const auto& n = pts.normals[i];
            f << p.x << ' ' << p.y << ' ' << p.z << ' '
              << n.x << ' ' << n.y << ' ' << n.z << '\n';
        }
    }
}

SampledPoints load_points(const std::string& path, double* out_sampling_radius) {
    if (out_sampling_radius) *out_sampling_radius = -1.0;

    // Open in binary mode so tellg()/read() work for binary PLY.
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(f, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();

    // PLY file
    if (line == "ply") {
        enum class Fmt { ASCII, LE, BE } fmt = Fmt::ASCII;
        int n_verts = 0;

        // Track the 6 properties we care about (x y z nx ny nz):
        // store their index in the property list and their type.
        struct PropInfo { std::string name, type; int size; };
        std::vector<PropInfo> props;
        bool in_vertex = false;

        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            std::string tok; ss >> tok;
            if (tok == "comment") {
                std::string key; ss >> key;
                if (key == "sampling_radius" && out_sampling_radius) {
                    double r; if (ss >> r) *out_sampling_radius = r;
                }
            } else if (tok == "format") {
                std::string s; ss >> s;
                if (s == "binary_little_endian") fmt = Fmt::LE;
                else if (s == "binary_big_endian") fmt = Fmt::BE;
            } else if (tok == "element") {
                std::string name; int cnt; ss >> name >> cnt;
                in_vertex = (name == "vertex");
                if (in_vertex) n_verts = cnt;
            } else if (tok == "property" && in_vertex) {
                std::string type, name; ss >> type >> name;
                if (type != "list") {
                    int sz = 0;
                    if (type=="char"||type=="int8"||type=="uchar"||type=="uint8") sz=1;
                    else if (type=="short"||type=="int16"||type=="ushort"||type=="uint16") sz=2;
                    else if (type=="int"||type=="int32"||type=="uint"||type=="uint32"||
                             type=="float"||type=="float32") sz=4;
                    else if (type=="double"||type=="float64") sz=8;
                    props.push_back({name, type, sz});
                }
            } else if (tok == "end_header") {
                break;
            }
        }
        // Stream is now at first data byte.

        // Map property name → index in props[]
        auto find_prop = [&](const std::string& name) -> int {
            for (int i = 0; i < (int)props.size(); i++)
                if (props[i].name == name) return i;
            return -1;
        };
        int xi=find_prop("x"), yi=find_prop("y"), zi=find_prop("z");
        int nxi=find_prop("nx"), nyi=find_prop("ny"), nzi=find_prop("nz");

        SampledPoints pts;
        pts.positions.reserve(n_verts);
        pts.normals.reserve(n_verts);

        bool is_binary = (fmt != Fmt::ASCII);
        bool be = (fmt == Fmt::BE);

        if (is_binary) {
            // Compute stride and per-property offsets.
            int stride = 0;
            std::vector<int> offsets;
            for (auto& p : props) { offsets.push_back(stride); stride += p.size; }

            // Helper: read one double from a buffer position.
            auto rd = [&](const char* buf, int idx) -> double {
                if (idx < 0) return 0.0;
                const char* p = buf + offsets[idx];
                const std::string& type = props[idx].type;
                if (type=="double"||type=="float64") {
                    uint64_t u; memcpy(&u,p,8);
                    if (be) u=__builtin_bswap64(u);
                    double v; memcpy(&v,&u,8); return v;
                }
                if (type=="float"||type=="float32") {
                    uint32_t u; memcpy(&u,p,4);
                    if (be) u=__builtin_bswap32(u);
                    float v; memcpy(&v,&u,4); return v;
                }
                // integer types: treat as signed
                if (props[idx].size == 4) {
                    uint32_t u; memcpy(&u,p,4);
                    if (be) u=__builtin_bswap32(u);
                    return static_cast<int32_t>(u);
                }
                if (props[idx].size == 2) {
                    uint16_t u; memcpy(&u,p,2);
                    if (be) u=__builtin_bswap16(u);
                    return static_cast<int16_t>(u);
                }
                return static_cast<int8_t>(p[0]);
            };

            std::vector<char> buf(stride);
            for (int i = 0; i < n_verts; i++) {
                if (!f.read(buf.data(), stride)) break;
                pts.positions.push_back({rd(buf.data(),xi), rd(buf.data(),yi), rd(buf.data(),zi)});
                pts.normals.push_back({rd(buf.data(),nxi), rd(buf.data(),nyi), rd(buf.data(),nzi)});
            }
        } else {
            for (int i = 0; i < n_verts && std::getline(f, line); i++) {
                std::istringstream ss(line);
                Vec3 p, n;
                if (ss >> p.x >> p.y >> p.z >> n.x >> n.y >> n.z) {
                    pts.positions.push_back(p);
                    pts.normals.push_back(n);
                }
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
