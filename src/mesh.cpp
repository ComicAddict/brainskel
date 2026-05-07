#include "mesh.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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
