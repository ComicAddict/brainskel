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

// PLY point cloud with per-sample positions (x y z) and outward normals
// (nx ny nz) stored as double.  sampling_radius, if > 0, is written as a
// PLY comment so the file is self-describing for medial-axis reconstruction.
// binary=true → binary_little_endian (default); false → ASCII.
void save_points(const std::string& path, const SampledPoints& pts,
                 bool binary = true, double sampling_radius = -1.0);

// Load a PLY point cloud written by save_points.
// If out_sampling_radius is non-null and the file contains a
// "comment sampling_radius R" line, *out_sampling_radius is set to R;
// otherwise it is set to -1.
SampledPoints load_points(const std::string& path,
                          double* out_sampling_radius = nullptr);
