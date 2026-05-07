#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <cstdint>
#include <zlib.h>

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
// GIFTI loader (.gii)
// Handles the three standard encodings: GZipBase64Binary, Base64Binary, ASCII.
// Supported DataTypes: NIFTI_TYPE_FLOAT32, NIFTI_TYPE_FLOAT64 (coords)
//                      NIFTI_TYPE_INT32 (triangles)
// ---------------------------------------------------------------------------

namespace {

// Base64 decode table, built once at startup.
static const auto& b64_table() {
    static const auto T = []() {
        std::array<int, 256> t;
        t.fill(-1);
        for (int i = 0; i < 26; i++) { t['A' + i] = i;      t['a' + i] = 26 + i; }
        for (int i = 0; i < 10; i++)   t['0' + i] = 52 + i;
        t['+'] = 62; t['/'] = 63; t['='] = 0;
        return t;
    }();
    return T;
}

static std::vector<uint8_t> base64_decode(const std::string& s) {
    const auto& T = b64_table();
    std::vector<uint8_t> out;
    out.reserve(s.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : s) {
        if (T[c] < 0) continue;
        val = (val << 6) + T[c];
        bits += 6;
        if (bits >= 0) { out.push_back((val >> bits) & 0xFF); bits -= 8; }
    }
    return out;
}

// Decompresses a zlib stream (the "GZipBase64Binary" GIFTI encoding is
// actually zlib-deflate with the standard 0x78 header, not raw gzip).
// inflateInit2 with windowBits = 47 auto-detects both zlib and gzip.
static std::vector<uint8_t> zlib_inflate(const std::vector<uint8_t>& in) {
    z_stream s{};
    if (inflateInit2(&s, 47) != Z_OK)
        throw std::runtime_error("zlib init failed");

    s.avail_in  = static_cast<uInt>(in.size());
    s.next_in   = const_cast<Bytef*>(in.data());

    std::vector<uint8_t> out;
    uint8_t buf[65536];
    int ret;
    do {
        s.avail_out = sizeof(buf);
        s.next_out  = buf;
        ret = inflate(&s, Z_NO_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END)
            { inflateEnd(&s); throw std::runtime_error("zlib inflate error"); }
        out.insert(out.end(), buf, buf + sizeof(buf) - s.avail_out);
    } while (ret != Z_STREAM_END);

    inflateEnd(&s);
    return out;
}

// Little/big-endian readers — GIFTI specifies the endianness per DataArray.
static float    rd_f32(const uint8_t* p, bool be) {
    uint32_t u = be
        ? (uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|p[3]
        :  uint32_t(p[0])     |(uint32_t(p[1])<<8) |(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
    float f; std::memcpy(&f, &u, 4); return f;
}
static double   rd_f64(const uint8_t* p, bool be) {
    uint64_t u = be
        ? (uint64_t(p[0])<<56)|(uint64_t(p[1])<<48)|(uint64_t(p[2])<<40)|(uint64_t(p[3])<<32)
         |(uint64_t(p[4])<<24)|(uint64_t(p[5])<<16)|(uint64_t(p[6])<<8) | p[7]
        :  uint64_t(p[0])     |(uint64_t(p[1])<<8) |(uint64_t(p[2])<<16)|(uint64_t(p[3])<<24)
         |(uint64_t(p[4])<<32)|(uint64_t(p[5])<<40)|(uint64_t(p[6])<<48)|(uint64_t(p[7])<<56);
    double d; std::memcpy(&d, &u, 8); return d;
}
static int32_t  rd_i32(const uint8_t* p, bool be) {
    uint32_t u = be
        ? (uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|p[3]
        :  uint32_t(p[0])     |(uint32_t(p[1])<<8) |(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
    int32_t i; std::memcpy(&i, &u, 4); return i;
}

// Extract the value of an XML attribute from a tag string, e.g.
// attr(tag, "Intent") -> "NIFTI_INTENT_POINTSET"
static std::string xml_attr(const std::string& tag, const std::string& name) {
    auto pos = tag.find(name + "=\"");
    if (pos == std::string::npos) return {};
    pos += name.size() + 2;
    auto end = tag.find('"', pos);
    return end != std::string::npos ? tag.substr(pos, end - pos) : std::string{};
}

} // anonymous namespace

static Mesh load_gifti(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::string xml((std::istreambuf_iterator<char>(f)), {});

    Mesh m;

    size_t pos = 0;
    while (true) {
        size_t arr = xml.find("<DataArray", pos);
        if (arr == std::string::npos) break;

        size_t gt = xml.find('>', arr);
        if (gt == std::string::npos) break;
        const std::string tag = xml.substr(arr, gt - arr);

        const std::string intent = xml_attr(tag, "Intent");
        const std::string dtype  = xml_attr(tag, "DataType");
        const std::string enc    = xml_attr(tag, "Encoding");
        const bool big_endian    = (xml_attr(tag, "Endian") == "BigEndian");
        const int dim0 = std::stoi(xml_attr(tag, "Dim0"));
        const int dim1 = std::stoi(xml_attr(tag, "Dim1"));

        // Locate <Data>...</Data> within this DataArray block (before next one).
        size_t next_arr  = xml.find("<DataArray", gt + 1);
        size_t data_open = xml.find("<Data>",     gt + 1);
        if (data_open == std::string::npos ||
            (next_arr != std::string::npos && data_open > next_arr)) {
            pos = gt + 1;
            continue;
        }
        data_open += 6;  // skip "<Data>"
        size_t data_close = xml.find("</Data>", data_open);
        if (data_close == std::string::npos) break;

        // Decode the raw bytes.
        std::vector<uint8_t> bytes;

        if (enc == "GZipBase64Binary" || enc == "Base64Binary") {
            std::string b64;
            b64.reserve(data_close - data_open);
            for (size_t i = data_open; i < data_close; i++)
                if (!std::isspace(static_cast<unsigned char>(xml[i])))
                    b64 += xml[i];
            bytes = base64_decode(b64);
            if (enc == "GZipBase64Binary")
                bytes = zlib_inflate(bytes);
        } else if (enc == "ASCII") {
            // Tokenise and convert to binary on the fly via the numeric paths below.
            // We set bytes empty and handle ASCII as a special case.
        } else {
            throw std::runtime_error("Unsupported GIFTI encoding: " + enc);
        }

        // ---- NIFTI_INTENT_POINTSET → vertex coordinates ----
        if (intent == "NIFTI_INTENT_POINTSET") {
            m.vertices.reserve(m.vertices.size() + dim0);
            if (enc == "ASCII") {
                std::istringstream ss(xml.substr(data_open, data_close - data_open));
                for (int i = 0; i < dim0; i++) {
                    Vec3 v{};
                    ss >> v.x >> v.y >> v.z;
                    m.vertices.push_back(v);
                }
            } else {
                const bool f64 = (dtype == "NIFTI_TYPE_FLOAT64");
                const int bpv  = f64 ? 8 : 4;
                for (int i = 0; i < dim0; i++) {
                    const uint8_t* p = bytes.data() + i * dim1 * bpv;
                    m.vertices.push_back(f64
                        ? Vec3{rd_f64(p,big_endian), rd_f64(p+8,big_endian), rd_f64(p+16,big_endian)}
                        : Vec3{rd_f32(p,big_endian), rd_f32(p+4,big_endian), rd_f32(p+8,big_endian)});
                }
            }

        // ---- NIFTI_INTENT_TRIANGLE → triangle connectivity ----
        } else if (intent == "NIFTI_INTENT_TRIANGLE") {
            m.triangles.reserve(m.triangles.size() + dim0);
            if (enc == "ASCII") {
                std::istringstream ss(xml.substr(data_open, data_close - data_open));
                for (int i = 0; i < dim0; i++) {
                    std::array<int,3> tri{};
                    ss >> tri[0] >> tri[1] >> tri[2];
                    m.triangles.push_back(tri);
                }
            } else {
                for (int i = 0; i < dim0; i++) {
                    const uint8_t* p = bytes.data() + i * 3 * 4;
                    m.triangles.push_back(std::array<int,3>{rd_i32(p,big_endian), rd_i32(p+4,big_endian), rd_i32(p+8,big_endian)});
                }
            }
        }

        pos = data_close + 7;
    }

    if (m.vertices.empty())
        throw std::runtime_error("No vertex data found in GIFTI: " + path);
    if (m.triangles.empty())
        throw std::runtime_error("No triangle data found in GIFTI: " + path);

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
    if (ext == "gii") return load_gifti(path);
    throw std::runtime_error("Unsupported mesh format: " + ext);
}

// ---------------------------------------------------------------------------
// weld_vertices
// ---------------------------------------------------------------------------

void weld_vertices(std::vector<Vec3>& vertices,
                   std::vector<std::vector<int>>& faces,
                   double tol) {
    int N = static_cast<int>(vertices.size());
    if (N == 0) return;

    // Grid cell = tol.  Each vertex is looked up in a 3×3×3 neighbourhood
    // of cells so bin-boundary artefacts are impossible.
    double tol2 = tol * tol;

    Vec3 bmin = vertices[0];
    for (const auto& v : vertices) {
        bmin.x = std::min(bmin.x, v.x);
        bmin.y = std::min(bmin.y, v.y);
        bmin.z = std::min(bmin.z, v.z);
    }

    using Key = std::tuple<long long, long long, long long>;
    auto make_key = [&](const Vec3& v, long long dx = 0, long long dy = 0, long long dz = 0) {
        return Key{
            static_cast<long long>(std::floor((v.x - bmin.x) / tol)) + dx,
            static_cast<long long>(std::floor((v.y - bmin.y) / tol)) + dy,
            static_cast<long long>(std::floor((v.z - bmin.z) / tol)) + dz
        };
    };

    // Grid maps a cell key to a list of representative vertex indices
    // (in new_verts).  Multiple representatives per cell are rare but
    // handled correctly.
    std::map<Key, std::vector<int>> grid;

    std::vector<int>  remap(N, -1);
    std::vector<Vec3> new_verts;
    new_verts.reserve(N);

    for (int i = 0; i < N; i++) {
        const Vec3& p = vertices[i];
        auto [kx, ky, kz] = make_key(p);

        int found = -1;
        for (long long dx = -1; dx <= 1 && found < 0; dx++)
        for (long long dy = -1; dy <= 1 && found < 0; dy++)
        for (long long dz = -1; dz <= 1 && found < 0; dz++) {
            auto it = grid.find({kx + dx, ky + dy, kz + dz});
            if (it == grid.end()) continue;
            for (int j : it->second) {
                Vec3 d = p - new_verts[j];
                if (d.dot(d) <= tol2) { found = j; break; }
            }
        }

        if (found >= 0) {
            remap[i] = found;
        } else {
            remap[i] = static_cast<int>(new_verts.size());
            grid[{kx, ky, kz}].push_back(remap[i]);
            new_verts.push_back(p);
        }
    }

    vertices = std::move(new_verts);

    // Remap face indices and remove faces that become degenerate.
    for (auto& face : faces) {
        for (int& idx : face) idx = remap[idx];
        // Remove consecutive duplicates (handles the wrapped edge too).
        face.erase(std::unique(face.begin(), face.end()), face.end());
        if (face.size() >= 2 && face.front() == face.back())
            face.pop_back();
    }
    faces.erase(
        std::remove_if(faces.begin(), faces.end(),
                       [](const std::vector<int>& f) { return f.size() < 3; }),
        faces.end());
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
