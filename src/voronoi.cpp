#include "voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "mesh.hpp"
#include "voro++.hh"

// ---------------------------------------------------------------------------
// compute_medial_axis
// ---------------------------------------------------------------------------

MedialAxisMesh compute_medial_axis(const SampledPoints& pts, double epsilon,
                                   bool merge_duplicates) {
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

    // Grid block counts: aim for ~8 particles per block.
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

    // ---- Collect particle descriptors serially ----------------------------
    // We need (ijk, q, i, j, k) to call voro_compute::compute_cell later.
    // The loop itself is not thread-safe, so this pass is serial.
    struct Particle { int id, ijk, q, ci, cj, ck; };
    std::vector<Particle> particles;
    particles.reserve(n_total);

    {
        voro::c_loop_all vl(con);
        if (vl.start()) do {
            particles.push_back({vl.pid(), vl.ijk, vl.q, vl.i, vl.j, vl.k});
        } while (vl.inc());
    }

    // ---- Parallel Voronoi cell computation --------------------------------
    //
    // voro_compute<container> owns all mutable working state (mask, queue).
    // The container's particle arrays are read-only at this point, so
    // multiple voro_compute instances can safely share the same container.
    //
    // For a non-periodic container voro++ initialises its internal
    // voro_compute with mask dims == grid dims (gnx, gny, gnz).
    // We replicate that here for each per-thread instance.
    //
    // Each thread accumulates into a private MedialAxisMesh; threads merge
    // results afterward (no locks needed during the parallel section).

    int n_threads = omp_get_max_threads();
    std::vector<MedialAxisMesh> thread_meshes(n_threads);

    #pragma omp parallel default(none) \
        shared(particles, con, thread_meshes, N, gnx, gny, gnz)
    {
        int tid = omp_get_thread_num();
        MedialAxisMesh& local = thread_meshes[tid];

        // Per-thread Voronoi compute engine.
        voro::voro_compute<voro::container> local_vc(con, gnx, gny, gnz);
        voro::voronoicell_neighbor vc;

        // Reusable scratch vectors (avoid repeated allocations).
        std::vector<int>    neigh, fv;
        std::vector<double> v;

        #pragma omp for schedule(dynamic, 32)
        for (int idx = 0; idx < static_cast<int>(particles.size()); idx++) {
            const Particle& pi = particles[idx];
            if (pi.id >= N) continue;  // skip outside-displaced points

            if (!local_vc.compute_cell(vc, pi.ijk, pi.q, pi.ci, pi.cj, pi.ck))
                continue;

            vc.neighbors(neigh);
            vc.vertices(con.p[pi.ijk][con.ps * pi.q],
                        con.p[pi.ijk][con.ps * pi.q + 1],
                        con.p[pi.ijk][con.ps * pi.q + 2], v);
            vc.face_vertices(fv);

            int fi = 0;
            for (int f = 0; f < static_cast<int>(neigh.size()); f++) {
                int n_verts = fv[fi++];
                int nb = neigh[f];

                // Outer cells are skipped at the top, so pi.id is always an
                // inside-displaced point (pi.id ∈ [0, N)).  Decide per face:
                //   inner–inner   keep, but record once (smaller id wins)
                //   inner–outer   keep unless nb is the *same sample's* outer
                //                 point (id == pi.id + N) — that's the "fake"
                //                 bisector right at the original surface
                //   bbox (nb<0)   skip
                bool keep = false;
                if (nb >= 0) {
                    if (nb < N) {
                        keep = (pi.id < nb);              // inner–inner
                    } else {
                        keep = (nb - N != pi.id);         // inner–outer
                    }
                }
                if (keep) {
                    int base = static_cast<int>(local.vertices.size());
                    for (int k = 0; k < n_verts; k++) {
                        int vi = fv[fi + k];
                        local.vertices.push_back(
                            {v[3*vi], v[3*vi+1], v[3*vi+2]});
                    }
                    std::vector<int> face_idx(n_verts);
                    std::iota(face_idx.begin(), face_idx.end(), base);
                    local.faces.push_back(std::move(face_idx));
                }

                fi += n_verts;
            }
        }
    }  // end parallel

    // ---- Merge per-thread results -----------------------------------------
    // Concatenate vertex lists; offset face indices of each shard.
    MedialAxisMesh result;
    for (auto& mesh : thread_meshes) {
        int base = static_cast<int>(result.vertices.size());
        result.vertices.insert(result.vertices.end(),
                               mesh.vertices.begin(), mesh.vertices.end());
        for (auto& face : mesh.faces) {
            result.faces.push_back(face);
            for (int& vi : result.faces.back()) vi += base;
        }
    }

    // ---- Weld near-duplicate vertices (optional) -------------------------
    // Tolerance absorbs inter-cell floating-point disagreement while staying
    // far below any real Voronoi edge length.
    if (merge_duplicates) {
        double weld_tol = std::max(extent * 1e-7, 1e-12);
        weld_vertices(result.vertices, result.faces, weld_tol);
    }

    return result;
}

// ---------------------------------------------------------------------------
// compute_medial_radii
// ---------------------------------------------------------------------------

std::vector<float> compute_medial_radii(const std::vector<Vec3>& ma_verts,
                                        const SampledPoints& pts) {
    int M = static_cast<int>(ma_verts.size());
    int N = static_cast<int>(pts.positions.size());
    std::vector<float> radii(M, 0.0f);
    if (M == 0 || N == 0) return radii;

    #pragma omp parallel for schedule(dynamic, 64) \
        default(none) shared(ma_verts, pts, radii, M, N)
    for (int i = 0; i < M; i++) {
        const Vec3& v = ma_verts[i];
        double best = std::numeric_limits<double>::max();
        for (int j = 0; j < N; j++) {
            const Vec3& p = pts.positions[j];
            double dx = v.x - p.x;
            double dy = v.y - p.y;
            double dz = v.z - p.z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best) best = d2;
        }
        radii[i] = static_cast<float>(std::sqrt(best));
    }
    return radii;
}
