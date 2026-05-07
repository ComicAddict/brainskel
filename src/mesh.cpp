#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Normals
// ---------------------------------------------------------------------------

void Mesh::compute_vertex_normals() {
    normals.assign(vertices.size(), Vec3{});
    for (auto& tri : triangles) {
        Vec3 e1 = vertices[tri[1]] - vertices[tri[0]];
        Vec3 e2 = vertices[tri[2]] - vertices[tri[0]];
        Vec3 fn = e1.cross(e2);  // weighted by area (no normalize)
        normals[tri[0]] += fn;
        normals[tri[1]] += fn;
        normals[tri[2]] += fn;
    }
    for (auto& n : normals)
        n = n.normalized();
}

double Mesh::surface_area() const {
    double area = 0.0;
    for (auto& tri : triangles) {
        Vec3 e1 = vertices[tri[1]] - vertices[tri[0]];
        Vec3 e2 = vertices[tri[2]] - vertices[tri[0]];
        area += 0.5 * e1.cross(e2).norm();
    }
    return area;
}

// ---------------------------------------------------------------------------
// OBJ loader
// ---------------------------------------------------------------------------

static Mesh load_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    Mesh m;
    std::vector<Vec3> file_normals;
    std::string line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "v") {
            Vec3 v;
            ss >> v.x >> v.y >> v.z;
            m.vertices.push_back(v);
        } else if (tok == "vn") {
            Vec3 n;
            ss >> n.x >> n.y >> n.z;
            file_normals.push_back(n.normalized());
        } else if (tok == "f") {
            // Accepts: v  v/vt  v//vn  v/vt/vn  (1-indexed, negative allowed)
            std::vector<int> vids;
            std::string fv;
            while (ss >> fv) {
                int vid = std::stoi(fv);  // only need position index
                if (vid < 0)
                    vid = (int)m.vertices.size() + vid;
                else
                    vid -= 1;
                vids.push_back(vid);
            }
            // Fan triangulation for non-triangle faces
            for (int i = 1; i + 1 < (int)vids.size(); i++)
                m.triangles.push_back({vids[0], vids[i], vids[i + 1]});
        }
    }

    m.compute_vertex_normals();
    return m;
}

// ---------------------------------------------------------------------------
// PLY loader (ASCII only)
// ---------------------------------------------------------------------------

static Mesh load_ply(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    int n_verts = 0, n_faces = 0;
    bool header = true;
    bool binary = false;

    while (header && std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;
        if (tok == "format") {
            std::string fmt; ss >> fmt;
            if (fmt != "ascii") {
                binary = true;
            }
        } else if (tok == "element") {
            std::string name; int count;
            ss >> name >> count;
            if (name == "vertex") n_verts = count;
            else if (name == "face") n_faces = count;
        } else if (tok == "end_header") {
            header = false;
        }
    }

    if (binary)
        throw std::runtime_error("Binary PLY not supported; convert to ASCII first.");

    Mesh m;
    m.vertices.reserve(n_verts);
    m.triangles.reserve(n_faces);

    for (int i = 0; i < n_verts; i++) {
        if (!std::getline(f, line)) break;
        std::istringstream ss(line);
        Vec3 v;
        ss >> v.x >> v.y >> v.z;
        m.vertices.push_back(v);
    }

    for (int i = 0; i < n_faces; i++) {
        if (!std::getline(f, line)) break;
        std::istringstream ss(line);
        int count;
        ss >> count;
        std::vector<int> vids(count);
        for (int j = 0; j < count; j++) ss >> vids[j];
        for (int j = 1; j + 1 < count; j++)
            m.triangles.push_back({vids[0], vids[j], vids[j + 1]});
    }

    m.compute_vertex_normals();
    return m;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Mesh load_mesh(const std::string& path) {
    auto ext_pos = path.rfind('.');
    if (ext_pos == std::string::npos)
        throw std::runtime_error("Cannot determine mesh format for: " + path);
    std::string ext = path.substr(ext_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "obj") return load_obj(path);
    if (ext == "ply") return load_ply(path);
    throw std::runtime_error("Unsupported mesh format: " + ext);
}

// ---------------------------------------------------------------------------
// weld_vertices
// ---------------------------------------------------------------------------

void weld_vertices(std::vector<Vec3>& vertices,
                   std::vector<std::vector<int>>& faces,
                   double tol) {
    int N = static_cast<int>(vertices.size());
    if (N == 0) return;

    // Grid cell = tol.  Each vertex is looked up in a 3×3×3 neighbourhood
    // of cells so bin-boundary artefacts are impossible.
    double tol2 = tol * tol;

    Vec3 bmin = vertices[0];
    for (const auto& v : vertices) {
        bmin.x = std::min(bmin.x, v.x);
        bmin.y = std::min(bmin.y, v.y);
        bmin.z = std::min(bmin.z, v.z);
    }

    using Key = std::tuple<long long, long long, long long>;
    auto make_key = [&](const Vec3& v, long long dx = 0, long long dy = 0, long long dz = 0) {
        return Key{
            static_cast<long long>(std::floor((v.x - bmin.x) / tol)) + dx,
            static_cast<long long>(std::floor((v.y - bmin.y) / tol)) + dy,
            static_cast<long long>(std::floor((v.z - bmin.z) / tol)) + dz
        };
    };

    // Grid maps a cell key to a list of representative vertex indices
    // (in new_verts).  Multiple representatives per cell are rare but
    // handled correctly.
    std::map<Key, std::vector<int>> grid;

    std::vector<int>  remap(N, -1);
    std::vector<Vec3> new_verts;
    new_verts.reserve(N);

    for (int i = 0; i < N; i++) {
        const Vec3& p = vertices[i];
        auto [kx, ky, kz] = make_key(p);

        int found = -1;
        for (long long dx = -1; dx <= 1 && found < 0; dx++)
        for (long long dy = -1; dy <= 1 && found < 0; dy++)
        for (long long dz = -1; dz <= 1 && found < 0; dz++) {
            auto it = grid.find({kx + dx, ky + dy, kz + dz});
            if (it == grid.end()) continue;
            for (int j : it->second) {
                Vec3 d = p - new_verts[j];
                if (d.dot(d) <= tol2) { found = j; break; }
            }
        }

        if (found >= 0) {
            remap[i] = found;
        } else {
            remap[i] = static_cast<int>(new_verts.size());
            grid[{kx, ky, kz}].push_back(remap[i]);
            new_verts.push_back(p);
        }
    }

    vertices = std::move(new_verts);

    // Remap face indices and remove faces that become degenerate.
    for (auto& face : faces) {
        for (int& idx : face) idx = remap[idx];
        // Remove consecutive duplicates (handles the wrapped edge too).
        face.erase(std::unique(face.begin(), face.end()), face.end());
        if (face.size() >= 2 && face.front() == face.back())
            face.pop_back();
    }
    faces.erase(
        std::remove_if(faces.begin(), faces.end(),
                       [](const std::vector<int>& f) { return f.size() < 3; }),
        faces.end());
}

void save_obj_polygons(const std::string& path,
                       const std::vector<Vec3>& vertices,
                       const std::vector<std::vector<int>>& faces) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f.precision(10);
    for (auto& v : vertices)
        f << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';

    for (auto& face : faces) {
        f << 'f';
        for (int idx : face)
            f << ' ' << (idx + 1);  // OBJ is 1-indexed
        f << '\n';
    }
}
