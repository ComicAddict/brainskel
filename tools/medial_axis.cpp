#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mesh.hpp"
#include "sampling.hpp"
#include "voronoi.hpp"

static void usage(const char* argv0) {
    std::cerr <<
        "Usage: " << argv0 << " <input.(obj|ply)> <output.obj>\n"
        "                  [--radius R]    minimum sample distance (default: auto)\n"
        "                  [--epsilon E]   displacement for inside/outside points\n"
        "                                  (default: radius/10)\n"
        "                  [--seed S]      RNG seed (default: 42)\n"
        "                  [--points P]    pre-sampled points file (skips sampling)\n"
        "\n"
        "Computes a Voronoi-based medial axis approximation from a closed surface\n"
        "mesh.  Writes the result as a polygon OBJ.\n"
        "\n"
        "Pipeline:\n"
        "  1. Poisson-disk sample the surface.\n"
        "  2. Displace each sample by epsilon inward (inside) and outward (outside).\n"
        "  3. Compute 3-D Voronoi tessellation of all displaced points.\n"
        "  4. Retain only faces whose two generating points are both 'inside'.\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    std::string input   = argv[1];
    std::string output  = argv[2];

    double radius  = -1.0;
    double epsilon = -1.0;
    unsigned int seed = 42;
    std::string points_file;

    for (int i = 3; i < argc; i++) {
        std::string flag = argv[i];
        if ((flag == "--radius"  || flag == "-r") && i + 1 < argc)
            radius = std::stod(argv[++i]);
        else if ((flag == "--epsilon" || flag == "-e") && i + 1 < argc)
            epsilon = std::stod(argv[++i]);
        else if ((flag == "--seed"    || flag == "-s") && i + 1 < argc)
            seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if ((flag == "--points"  || flag == "-p") && i + 1 < argc)
            points_file = argv[++i];
        else { usage(argv[0]); return 1; }
    }

    try {
        SampledPoints pts;

        if (!points_file.empty()) {
            std::cerr << "Loading pre-sampled points: " << points_file << " ...\n";
            pts = load_points(points_file);
            std::cerr << "  " << pts.positions.size() << " points\n";
        } else {
            std::cerr << "Loading mesh: " << input << " ...\n";
            Mesh mesh = load_mesh(input);
            std::cerr << "  " << mesh.vertices.size() << " vertices, "
                      << mesh.triangles.size() << " triangles\n";

            if (radius <= 0.0) {
                double area = mesh.surface_area();
                radius = std::sqrt(area / (M_PI * 10000.0 / 4.0));
                std::cerr << "  Auto radius: " << radius
                          << " (surface area = " << area << ")\n";
            }

            std::cerr << "Sampling (min radius=" << radius
                      << ", seed=" << seed << ") ...\n";
            pts = poisson_disk_sample(mesh, radius, seed);
            std::cerr << "  " << pts.positions.size() << " samples\n";
        }

        if (epsilon <= 0.0) {
            // Epsilon should be small relative to sample spacing.
            if (radius > 0.0)
                epsilon = radius * 0.05;
            else {
                // Estimate from point cloud spread.
                Vec3 bmin = pts.positions[0], bmax = pts.positions[0];
                for (const auto& p : pts.positions) {
                    bmin.x = std::min(bmin.x, p.x); bmax.x = std::max(bmax.x, p.x);
                    bmin.y = std::min(bmin.y, p.y); bmax.y = std::max(bmax.y, p.y);
                    bmin.z = std::min(bmin.z, p.z); bmax.z = std::max(bmax.z, p.z);
                }
                double extent = std::max({bmax.x - bmin.x,
                                          bmax.y - bmin.y,
                                          bmax.z - bmin.z});
                epsilon = extent * 0.001;
            }
            std::cerr << "  Auto epsilon: " << epsilon << "\n";
        }

        std::cerr << "Computing medial axis (epsilon=" << epsilon << ") ...\n";
        MedialAxisMesh ma = compute_medial_axis(pts, epsilon);
        std::cerr << "  " << ma.vertices.size() << " vertices, "
                  << ma.faces.size() << " faces\n";

        std::cerr << "Writing: " << output << " ...\n";
        save_obj_polygons(output, ma.vertices, ma.faces);
        std::cerr << "Done.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
