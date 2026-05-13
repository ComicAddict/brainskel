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
//   Voronoi tessellation is computed over all 2*N displaced points.  For
//   each Voronoi face we apply the following keep/remove rule based on the
//   two generating points:
//     inner – inner                                 → KEEP
//     outer – outer                                 → REMOVE
//     same-sample inner / outer (perpendicular
//       bisector lying right at the surface)        → REMOVE
//     inner / outer from *different* samples        → KEEP
//
// epsilon: displacement magnitude — should be much smaller than min_radius
//          used during sampling (e.g. min_radius / 20).
// merge_duplicates: if true (default), weld near-coincident Voronoi vertices
//          that were independently computed in neighbouring cells, producing
//          a connected mesh.  Set to false to keep the raw per-face vertex
//          soup (useful for debugging or for downstream tools that prefer it).
MedialAxisMesh compute_medial_axis(const SampledPoints& pts, double epsilon,
                                   bool merge_duplicates = true);

// For each vertex of the medial axis mesh, return its distance to the nearest
// original surface sample point (pts.positions).  This approximates the local
// feature size — the radius of the maximal inscribed sphere at that point.
// Parallelised with OpenMP; complexity O(V_ma × N_samples / threads).
std::vector<float> compute_medial_radii(const std::vector<Vec3>& ma_vertices,
                                        const SampledPoints& pts);
