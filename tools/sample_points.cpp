#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mesh.hpp"
#include "sampling.hpp"

static void usage(const char* argv0) {
    std::cerr <<
        "Usage: " << argv0 << " <input.(obj|ply|gii)> <output.ply>\n"
        "                   [--radius R]   minimum sample distance (default: auto)\n"
        "                   [--seed S]     RNG seed (default: 42)\n"
        "                   [--ascii]      write ASCII PLY (default: binary)\n"
        "\n"
        "Samples points on the mesh surface using Poisson-disk sampling.\n"
        "Output is a PLY point cloud (x y z nx ny nz) importable in Blender.\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    std::string input  = argv[1];
    std::string output = argv[2];

    double radius = -1.0;
    unsigned int seed = 42;
    bool binary = true;

    for (int i = 3; i < argc; i++) {
        std::string flag = argv[i];
        if ((flag == "--radius" || flag == "-r") && i + 1 < argc)
            radius = std::stod(argv[++i]);
        else if ((flag == "--seed" || flag == "-s") && i + 1 < argc)
            seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (flag == "--ascii")
            binary = false;
        else { usage(argv[0]); return 1; }
    }

    try {
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
        SampledPoints pts = poisson_disk_sample(mesh, radius, seed);
        std::cerr << "  " << pts.positions.size() << " samples\n";

        std::cerr << "Writing: " << output
                  << " (" << (binary ? "binary" : "ascii") << " PLY) ...\n";
        save_points(output, pts, binary);
        std::cerr << "Done.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
