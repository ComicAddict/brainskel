#pragma once
#include <array>
#include <cmath>
#include <string>
#include <vector>

struct Vec3 {
    double x{}, y{}, z{};

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) {
        x += o.x; y += o.y; z += o.z; return *this;
    }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double norm2() const { return dot(*this); }
    double norm() const { return std::sqrt(norm2()); }
    Vec3 normalized() const {
        double n = norm();
        return n > 1e-12 ? *this / n : Vec3{};
    }
};

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;              // per-vertex outward normals
    std::vector<std::array<int, 3>> triangles;

    void compute_vertex_normals();
    double surface_area() const;
};

// Auto-detects format by extension (.obj / .ply).
Mesh load_mesh(const std::string& path);

// Write a polygon soup as OBJ (faces may be polygons, not only triangles).
void save_obj_polygons(const std::string& path,
                       const std::vector<Vec3>& vertices,
                       const std::vector<std::vector<int>>& faces);

// Write a polygon soup as PLY.  binary=true → binary_little_endian (default).
// Vertex coords stored as float32; face indices as int32 with uint8 count.
void save_ply_polygons(const std::string& path,
                       const std::vector<Vec3>& vertices,
                       const std::vector<std::vector<int>>& faces,
                       bool binary = true);

// Merge vertices whose Euclidean distance is <= tol, update face indices,
// and remove any faces that become degenerate (< 3 distinct vertices).
// Uses a spatial grid so complexity is O(N) in the common case.
void weld_vertices(std::vector<Vec3>& vertices,
                   std::vector<std::vector<int>>& faces,
                   double tol);
