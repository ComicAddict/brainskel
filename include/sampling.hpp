#pragma once
#include "mesh.hpp"
#include <string>
#include <vector>

struct SampledPoints {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;  // outward surface normals at each sample
};

// Poisson-disk sampling on the mesh surface.
// min_radius: minimum Euclidean distance between any two samples.
// seed: RNG seed for reproducibility.
SampledPoints poisson_disk_sample(const Mesh& mesh, double min_radius,
                                   unsigned int seed = 42);

// Text format: one sample per line "x y z nx ny nz"
void save_points(const std::string& path, const SampledPoints& pts);
SampledPoints load_points(const std::string& path);
