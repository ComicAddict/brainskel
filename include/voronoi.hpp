#pragma once
#include "mesh.hpp"
#include "sampling.hpp"
#include <vector>

struct MedialAxisMesh {
    std::vector<Vec3> vertices;
    std::vector<std::vector<int>> faces;  // polygon faces (not triangulated)
};

// Compute a Voronoi-based medial axis approximation.
//
// Algorithm:
//   Each surface sample is displaced by `epsilon` along its normal to produce
//   an "inside" point (inward) and an "outside" point (outward).  A 3-D
//   Voronoi tessellation is computed over all 2*N displaced points.  Any
//   Voronoi face whose two generating points are both "inside" points is kept
//   as part of the medial axis; faces touching an "outside" point are
//   discarded.
//
// epsilon: displacement magnitude — should be much smaller than min_radius
//          used during sampling (e.g. min_radius / 10).
MedialAxisMesh compute_medial_axis(const SampledPoints& pts, double epsilon);
