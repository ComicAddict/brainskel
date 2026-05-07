#include "voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "mesh.hpp"
#include "voro++.hh"

// ---------------------------------------------------------------------------
// compute_medial_axis
// ---------------------------------------------------------------------------

MedialAxisMesh compute_medial_axis(const SampledPoints& pts, double epsilon) {
    int N = static_cast<int>(pts.positions.size());
    if (N == 0) throw std::runtime_error("No input points.");

    // IDs 0..N-1  → inside-displaced  (point - epsilon * normal)
    // IDs N..2N-1 → outside-displaced (point + epsilon * normal)
    std::vector<Vec3> all(2 * N);
    for (int i = 0; i < N; i++) {
        all[i]     = pts.positions[i] - pts.normals[i] * epsilon;
        all[i + N] = pts.positions[i] + pts.normals[i] * epsilon;
    }

    // Bounding box with a small padding.
    Vec3 bmin = all[0], bmax = all[0];
    for (const auto& p : all) {
        bmin.x = std::min(bmin.x, p.x); bmax.x = std::max(bmax.x, p.x);
        bmin.y = std::min(bmin.y, p.y); bmax.y = std::max(bmax.y, p.y);
        bmin.z = std::min(bmin.z, p.z); bmax.z = std::max(bmax.z, p.z);
    }
    double extent = std::max({bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z});
    double pad    = extent * 0.02;
    bmin.x -= pad; bmin.y -= pad; bmin.z -= pad;
    bmax.x += pad; bmax.y += pad; bmax.z += pad;

    // Choose grid block counts: aim for ~8 particles per block.
    int n_total = 2 * N;
    double cbrt_n = std::cbrt(static_cast<double>(n_total) / 8.0);
    double dx = bmax.x - bmin.x, dy = bmax.y - bmin.y, dz = bmax.z - bmin.z;
    double max_d = std::max({dx, dy, dz});
    int gnx = std::max(1, static_cast<int>(cbrt_n * dx / max_d));
    int gny = std::max(1, static_cast<int>(cbrt_n * dy / max_d));
    int gnz = std::max(1, static_cast<int>(cbrt_n * dz / max_d));

    voro::container con(bmin.x, bmax.x,
                        bmin.y, bmax.y,
                        bmin.z, bmax.z,
                        gnx, gny, gnz,
                        false, false, false,  // non-periodic
                        8);

    for (int i = 0; i < n_total; i++)
        con.put(i, all[i].x, all[i].y, all[i].z);

    // Collect qualifying Voronoi faces as a raw vertex soup.  The same
    // geometric vertex will appear once per adjacent cell with minutely
    // different floating-point values; weld_vertices() fuses them afterward.
    MedialAxisMesh result;

    voro::c_loop_all vl(con);
    voro::voronoicell_neighbor vc;

    if (!vl.start()) return result;

    do {
        if (!con.compute_cell(vc, vl)) continue;

        int id = vl.pid();
        if (id >= N) continue;  // only process inside points

        double cx = vl.x(), cy = vl.y(), cz = vl.z();

        std::vector<int> neigh;
        vc.neighbors(neigh);  // one entry per face

        std::vector<double> v;
        vc.vertices(cx, cy, cz, v);  // absolute positions, flat [x0,y0,z0,...]

        std::vector<int> fv;
        vc.face_vertices(fv);  // [n0, vi0, vi1, ..., n1, ...]

        int fi = 0;
        for (int f = 0; f < static_cast<int>(neigh.size()); f++) {
            int n_verts = fv[fi++];
            int nb = neigh[f];

            // Keep faces shared by two inside points.
            // Record only from the cell with the smaller ID to avoid duplicates.
            if (nb >= 0 && nb < N && id < nb) {
                int base = static_cast<int>(result.vertices.size());
                for (int k = 0; k < n_verts; k++) {
                    int vi = fv[fi + k];
                    result.vertices.push_back({v[3*vi], v[3*vi+1], v[3*vi+2]});
                }
                std::vector<int> face_idx(n_verts);
                std::iota(face_idx.begin(), face_idx.end(), base);
                result.faces.push_back(std::move(face_idx));
            }

            fi += n_verts;
        }
    } while (vl.inc());

    // Weld: merge vertices within a tolerance that absorbs the floating-point
    // differences between adjacent cells, while staying far below any real
    // geometric feature.  1e-7 * extent is ~10^7 x larger than double-precision
    // noise yet ~10^5 x smaller than the minimum Voronoi edge for typical inputs.
    // Tolerance: large enough to absorb inter-cell fp disagreement on the same
    // Voronoi vertex (~1e-10 for brain-scale coords), far smaller than any
    // real Voronoi edge (typically >> epsilon).
    double weld_tol = std::max(extent * 1e-7, 1e-12);
    weld_vertices(result.vertices, result.faces, weld_tol);

    return result;
}
