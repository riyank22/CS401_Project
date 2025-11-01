#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <string>
#include <numeric>
#include <chrono>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iomanip> // For std::setprecision
#include <fstream>

// STB Image libraries
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

// Image struct with timers and JSON name helper
struct Image {
    std::string name, ext;
    int width, height, channels_in;
    int channels_out;
    unsigned char* input_host;
    unsigned char* output_host;

    float time_load_ms = 0.0f;
    float time_process_ms = 0.0f;
    float time_save_ms = 0.0f;

    std::string get_json_name() const {
        std::string full_name = name + ext;
        std::string safe_name;
        for (char c : full_name) {
            if (c == '\\') { safe_name += "\\\\"; }
            else if (c == '"') { safe_name += "\\\""; }
            else { safe_name += c; }
        }
        return safe_name;
    }
};

// ==================== KERNEL DEFINITIONS ====================
// (Paste your 27x27 GAUSSIAN_27x27 and 3x3 sobel kernels here)
const float GAUSSIAN_27x27[729] = {
    0.00070489f,    0.00075901f,    0.00081245f,    0.00086453f,    0.00091452f,    0.00096170f,    0.00100534f,    0.00104476f,    0.00107932f,    0.00110844f,    0.00113164f,    0.00114850f,    0.00115874f,    0.00116217f,    0.00115874f,    0.00114850f,    0.00113164f,    0.00110844f,    0.00107932f,    0.00104476f,    0.00100534f,    0.00096170f,    0.00091452f,    0.00086453f,    0.00081245f,    0.00075901f,    0.00070489f,
    0.00075901f,    0.00081728f,    0.00087482f,    0.00093090f,    0.00098473f,    0.00103552f,    0.00108251f,    0.00112496f,    0.00116217f,    0.00119353f,    0.00121851f,    0.00123667f,    0.00124769f,    0.00125139f,    0.00124769f,    0.00123667f,    0.00121851f,    0.00119353f,    0.00116217f,    0.00112496f,    0.00108251f,    0.00103552f,    0.00098473f,    0.00093090f,    0.00087482f,    0.00081728f,    0.00075901f,
    0.00081245f,    0.00087482f,    0.00093643f,    0.00099645f,    0.00105407f,    0.00110844f,    0.00115874f,    0.00120418f,    0.00124401f,    0.00127758f,    0.00130431f,    0.00132375f,    0.00133555f,    0.00133951f,    0.00133555f,    0.00132375f,    0.00130431f,    0.00127758f,    0.00124401f,    0.00120418f,    0.00115874f,    0.00110844f,    0.00105407f,    0.00099645f,    0.00093643f,    0.00087482f,    0.00081245f,
    0.00086453f,    0.00093090f,    0.00099645f,    0.00106033f,    0.00112164f,    0.00117949f,    0.00123302f,    0.00128136f,    0.00132375f,    0.00135947f,    0.00138792f,    0.00140860f,    0.00142116f,    0.00142537f,    0.00142116f,    0.00140860f,    0.00138792f,    0.00135947f,    0.00132375f,    0.00128136f,    0.00123302f,    0.00117949f,    0.00112164f,    0.00106033f,    0.00099645f,    0.00093090f,    0.00086453f,
    0.00091452f,    0.00098473f,    0.00105407f,    0.00112164f,    0.00118649f,    0.00124769f,    0.00130431f,    0.00135546f,    0.00140029f,    0.00143808f,    0.00146817f,    0.00149005f,    0.00150334f,    0.00150779f,    0.00150334f,    0.00149005f,    0.00146817f,    0.00143808f,    0.00140029f,    0.00135546f,    0.00130431f,    0.00124769f,    0.00118649f,    0.00112164f,    0.00105407f,    0.00098473f,    0.00091452f,
    0.00096170f,    0.00103552f,    0.00110844f,    0.00117949f,    0.00124769f,    0.00131205f,    0.00137159f,    0.00142537f,    0.00147252f,    0.00151226f,    0.00154391f,    0.00156691f,    0.00158088f,    0.00158557f,    0.00158088f,    0.00156691f,    0.00154391f,    0.00151226f,    0.00147252f,    0.00142537f,    0.00137159f,    0.00131205f,    0.00124769f,    0.00117949f,    0.00110844f,    0.00103552f,    0.00096170f,
    0.00100534f,    0.00108251f,    0.00115874f,    0.00123302f,    0.00130431f,    0.00137159f,    0.00143383f,    0.00149005f,    0.00153934f,    0.00158088f,    0.00161397f,    0.00163802f,    0.00165262f,    0.00165752f,    0.00165262f,    0.00163802f,    0.00161397f,    0.00158088f,    0.00153934f,    0.00149005f,    0.00143383f,    0.00137159f,    0.00130431f,    0.00123302f,    0.00115874f,    0.00108251f,    0.00100534f,
    0.00104476f,    0.00112496f,    0.00120418f,    0.00128136f,    0.00135546f,    0.00142537f,    0.00149005f,    0.00154848f,    0.00159970f,    0.00164287f,    0.00167725f,    0.00170225f,    0.00171742f,    0.00172251f,    0.00171742f,    0.00170225f,    0.00167725f,    0.00164287f,    0.00159970f,    0.00154848f,    0.00149005f,    0.00142537f,    0.00135546f,    0.00128136f,    0.00120418f,    0.00112496f,    0.00104476f,
    0.00107932f,    0.00116217f,    0.00124401f,    0.00132375f,    0.00140029f,    0.00147252f,    0.00153934f,    0.00159970f,    0.00165262f,    0.00169722f,    0.00173273f,    0.00175856f,    0.00177423f,    0.00177949f,    0.00177423f,    0.00175856f,    0.00173273f,    0.00169722f,    0.00165262f,    0.00159970f,    0.00153934f,    0.00147252f,    0.00140029f,    0.00132375f,    0.00124401f,    0.00116217f,    0.00107932f,
    0.00110844f,    0.00119353f,    0.00127758f,    0.00135947f,    0.00143808f,    0.00151226f,    0.00158088f,    0.00164287f,    0.00169722f,    0.00174302f,    0.00177949f,    0.00180601f,    0.00182211f,    0.00182751f,    0.00182211f,    0.00180601f,    0.00177949f,    0.00174302f,    0.00169722f,    0.00164287f,    0.00158088f,    0.00151226f,    0.00143808f,    0.00135947f,    0.00127758f,    0.00119353f,    0.00110844f,
    0.00113164f,    0.00121851f,    0.00130431f,    0.00138792f,    0.00146817f,    0.00154391f,    0.00161397f,    0.00167725f,    0.00173273f,    0.00177949f,    0.00181673f,    0.00184380f,    0.00186024f,    0.00186575f,    0.00186024f,    0.00184380f,    0.00181673f,    0.00177949f,    0.00173273f,    0.00167725f,    0.00161397f,    0.00154391f,    0.00146817f,    0.00138792f,    0.00130431f,    0.00121851f,    0.00113164f,
    0.00114850f,    0.00123667f,    0.00132375f,    0.00140860f,    0.00149005f,    0.00156691f,    0.00163802f,    0.00170225f,    0.00175856f,    0.00180601f,    0.00184380f,    0.00187128f,    0.00188796f,    0.00189356f,    0.00188796f,    0.00187128f,    0.00184380f,    0.00180601f,    0.00175856f,    0.00170225f,    0.00163802f,    0.00156691f,    0.00149005f,    0.00140860f,    0.00132375f,    0.00123667f,    0.00114850f,
    0.00115874f,    0.00124769f,    0.00133555f,    0.00142116f,    0.00150334f,    0.00158088f,    0.00165262f,    0.00171742f,    0.00177423f,    0.00182211f,    0.00186024f,    0.00188796f,    0.00190480f,    0.00191044f,    0.00190480f,    0.00188796f,    0.00186024f,    0.00182211f,    0.00177423f,    0.00171742f,    0.00165262f,    0.00158088f,    0.00150334f,    0.00142116f,    0.00133555f,    0.00124769f,    0.00115874f,
    0.00116217f,    0.00125139f,    0.00133951f,    0.00142537f,    0.00150779f,    0.00158557f,    0.00165752f,    0.00172251f,    0.00177949f,    0.00182751f,    0.00186575f,    0.00189356f,    0.00191044f,    0.00191610f,    0.00191044f,    0.00189356f,    0.00186575f,    0.00182751f,    0.00177949f,    0.00172251f,    0.00165752f,    0.00158557f,    0.00150779f,    0.00142537f,    0.00133951f,    0.00125139f,    0.00116217f,
    0.00115874f,    0.00124769f,    0.00133555f,    0.00142116f,    0.00150334f,    0.00158088f,    0.00165262f,    0.00171742f,    0.00177423f,    0.00182211f,    0.00186024f,    0.00188796f,    0.00190480f,    0.00191044f,    0.00190480f,    0.00188796f,    0.00186024f,    0.00182211f,    0.00177423f,    0.00171742f,    0.00165262f,    0.00158088f,    0.00150334f,    0.00142116f,    0.00133555f,    0.00124769f,    0.00115874f,
    0.00114850f,    0.00123667f,    0.00132375f,    0.00140860f,    0.00149005f,    0.00156691f,    0.00163802f,    0.00170225f,    0.00175856f,    0.00180601f,    0.00184380f,    0.00187128f,    0.00188796f,    0.00189356f,    0.00188796f,    0.00187128f,    0.00184380f,    0.00180601f,    0.00175856f,    0.00170225f,    0.00163802f,    0.00156691f,    0.00149005f,    0.00140860f,    0.00132375f,    0.00123667f,    0.00114850f,
    0.00113164f,    0.00121851f,    0.00130431f,    0.00138792f,    0.00146817f,    0.00154391f,    0.00161397f,    0.00167725f,    0.00173273f,    0.00177949f,    0.00181673f,    0.00184380f,    0.00186024f,    0.00186575f,    0.00186024f,    0.00184380f,    0.00181673f,    0.00177949f,    0.00173273f,    0.00167725f,    0.00161397f,    0.00154391f,    0.00146817f,    0.00138792f,    0.00130431f,    0.00121851f,    0.00113164f,
    0.00110844f,    0.00119353f,    0.00127758f,    0.00135947f,    0.00143808f,    0.00151226f,    0.00158088f,    0.00164287f,    0.00169722f,    0.00174302f,    0.00177949f,    0.00180601f,    0.00182211f,    0.00182751f,    0.00182211f,    0.00180601f,    0.00177949f,    0.00174302f,    0.00169722f,    0.00164287f,    0.00158088f,    0.00151226f,    0.00143808f,    0.00135947f,    0.00127758f,    0.00119353f,    0.00110844f,
    0.00107932f,    0.00116217f,    0.00124401f,    0.00132375f,    0.00140029f,    0.00147252f,    0.00153934f,    0.00159970f,    0.00165262f,    0.00169722f,    0.00173273f,    0.00175856f,    0.00177423f,    0.00177949f,    0.00177423f,    0.00175856f,    0.00173273f,    0.00169722f,    0.00165262f,    0.00159970f,    0.00153934f,    0.00147252f,    0.00140029f,    0.00132375f,    0.00124401f,    0.00116217f,    0.00107932f,
    0.00104476f,    0.00112496f,    0.00120418f,    0.00128136f,    0.00135546f,    0.00142537f,    0.00149005f,    0.00154848f,    0.00159970f,    0.00164287f,    0.00167725f,    0.00170225f,    0.00171742f,    0.00172251f,    0.00171742f,    0.00170225f,    0.00167725f,    0.00164287f,    0.00159970f,    0.00154848f,    0.00149005f,    0.00142537f,    0.00135546f,    0.00128136f,    0.00120418f,    0.00112496f,    0.00104476f,
    0.00100534f,    0.00108251f,    0.00115874f,    0.00123302f,    0.00130431f,    0.00137159f,    0.00143383f,    0.00149005f,    0.00153934f,    0.00158088f,    0.00161397f,    0.00163802f,    0.00165262f,    0.00165752f,    0.00165262f,    0.00163802f,    0.00161397f,    0.00158088f,    0.00153934f,    0.00149005f,    0.00143383f,    0.00137159f,    0.00130431f,    0.00123302f,    0.00115874f,    0.00108251f,    0.00100534f,
    0.00096170f,    0.00103552f,    0.00110844f,    0.00117949f,    0.00124769f,    0.00131205f,    0.00137159f,    0.00142537f,    0.00147252f,    0.00151226f,    0.00154391f,    0.00156691f,    0.00158088f,    0.00158557f,    0.00158088f,    0.00156691f,    0.00154391f,    0.00151226f,    0.00147252f,    0.00142537f,    0.00137159f,    0.00131205f,    0.00124769f,    0.00117949f,    0.00110844f,    0.00103552f,    0.00096170f,
    0.00091452f,    0.00098473f,    0.00105407f,    0.00112164f,    0.00118649f,    0.00124769f,    0.00130431f,    0.00135546f,    0.00140029f,    0.00143808f,    0.00146817f,    0.00149005f,    0.00150334f,    0.00150779f,    0.00150334f,    0.00149005f,    0.00146817f,    0.00143808f,    0.00140029f,    0.00135546f,    0.00130431f,    0.00124769f,    0.00118649f,    0.00112164f,    0.00105407f,    0.00098473f,    0.00091452f,
    0.00086453f,    0.00093090f,    0.00099645f,    0.00106033f,    0.00112164f,    0.00117949f,    0.00123302f,    0.00128136f,    0.00132375f,    0.00135947f,    0.00138792f,    0.00140860f,    0.00142116f,    0.00142537f,    0.00142116f,    0.00140860f,    0.00138792f,    0.00135947f,    0.00132375f,    0.00128136f,    0.00123302f,    0.00117949f,    0.00112164f,    0.00106033f,    0.00099645f,    0.00093090f,    0.00086453f,
    0.00081245f,    0.00087482f,    0.00093643f,    0.00099645f,    0.00105407f,    0.00110844f,    0.00115874f,    0.00120418f,    0.00124401f,    0.00127758f,    0.00130431f,    0.00132375f,    0.00133555f,    0.00133951f,    0.00133555f,    0.00132375f,    0.00130431f,    0.00127758f,    0.00124401f,    0.00120418f,    0.00115874f,    0.00110844f,    0.00105407f,    0.00099645f,    0.00093643f,    0.00087482f,    0.00081245f,
    0.00075901f,    0.00081728f,    0.00087482f,    0.00093090f,    0.00098473f,    0.00103552f,    0.00108251f,    0.00112496f,    0.00116217f,    0.00119353f,    0.00121851f,    0.00123667f,    0.00124769f,    0.00125139f,    0.00124769f,    0.00123667f,    0.00121851f,    0.00119353f,    0.00116217f,    0.00112496f,    0.00108251f,    0.00103552f,    0.00098473f,    0.00093090f,    0.00087482f,    0.00081728f,    0.00075901f,
    0.00070489f,    0.00075901f,    0.00081245f,    0.00086453f,    0.00091452f,    0.00096170f,    0.00100534f,    0.00104476f,    0.00107932f,    0.00110844f,    0.00113164f,    0.00114850f,    0.00115874f,    0.00116217f,    0.00115874f,    0.00114850f,    0.00113164f,    0.00110844f,    0.00107932f,    0.00104476f,    0.00100534f,    0.00096170f,    0.00091452f,    0.00086453f,    0.00081245f,    0.00075901f,    0.00070489f

};
const float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
const float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
// ========================================================

// ==================== CPU FILTER FUNCTIONS (OMP) ====================
inline int clamp(int val, int min_val, int max_val) {
    return std::min(std::max(val, min_val), max_val);
}
inline float clamp_f(float val, float min_val, float max_val) {
    return std::min(std::max(val, min_val), max_val);
}

void apply_grayscale(const unsigned char* in, unsigned char* out, int w, int h, int c_in) {
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx_in = (y * w + x) * c_in;
            int idx_out = y * w + x;
            float r = in[idx_in], g = in[idx_in + 1], b = in[idx_in + 2];
            out[idx_out] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

void apply_gaussian(const unsigned char* in, unsigned char* out, int w, int h, int c_in,
                    const float* kernel, int k_size) {
    int half = k_size / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int nx = clamp(x + kx, 0, w - 1);
                    int ny = clamp(y + ky, 0, h - 1);
                    int nIdx = (ny * w + nx) * c_in;
                    float k_val = kernel[(ky + half) * k_size + (kx + half)];
                    r_sum += k_val * in[nIdx];
                    g_sum += k_val * in[nIdx + 1];
                    b_sum += k_val * in[nIdx + 2];
                }
            }
            int outIdx = (y * w + x) * c_in;
            out[outIdx]     = static_cast<unsigned char>(clamp_f(r_sum, 0.0f, 255.0f));
            out[outIdx + 1] = static_cast<unsigned char>(clamp_f(g_sum, 0.0f, 255.0f));
            out[outIdx + 2] = static_cast<unsigned char>(clamp_f(b_sum, 0.0f, 255.0f));
        }
    }
}

void apply_sobel(const unsigned char* in, unsigned char* out, int w, int h, int c_in,
                 const float* kx_kernel, const float* ky_kernel, int k_size) {
    int half = k_size / 2;
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int nx = clamp(x + kx, 0, w - 1);
                    int ny = clamp(y + ky, 0, h - 1);
                    int idx = (ny * w + nx) * c_in;
                    float gray = 0.299f * in[idx] + 0.587f * in[idx + 1] + 0.114f * in[idx + 2];
                    gx += gray * kx_kernel[(ky + half) * k_size + (kx + half)];
                    gy += gray * ky_kernel[(ky + half) * k_size + (kx + half)];
                }
            }
            float mag = std::sqrt(gx * gx + gy * gy);
            out[y * w + x] = static_cast<unsigned char>(clamp_f(mag, 0.0f, 255.0f));
        }
    }
}
// ========================================================

int main(int argc, char** argv)
{
    // --- 1. Argument and Folder Setup ---
    if (argc < 4) {
        std::cerr << "Usage: ./ProjectCode_OMP <input_folder> <output_folder> <operation>\n";
        std::cerr << "Operations: grayscale | gaussian | sobel\n";
        return 1;
    }

    std::string folder = argv[1];
    std::string output_folder = argv[2];
    std::string op = argv[3];

    const int KERNEL_SIZE_GAUSSIAN = 27;
    const int KERNEL_SIZE_SOBEL = 3;

    int output_channels;
    if (op == "grayscale" || op == "sobel") { output_channels = 1; }
    else if (op == "gaussian") { output_channels = 3; }
    else { std::cerr << "Unknown operation: " << op << "\n"; return 1; }

    fs::create_directories(output_folder);

    std::vector<Image> images;

    // --- 2. Load all images from host memory (with timing) ---
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        Image img;
        img.name = entry.path().stem().string();
        img.ext = entry.path().extension().string();
        auto host_start = std::chrono::high_resolution_clock::now();
        int w, h, c;
        img.input_host = stbi_load(entry.path().c_str(), &w, &h, &c, 3);
        auto host_stop = std::chrono::high_resolution_clock::now();
        img.time_load_ms = std::chrono::duration<float, std::milli>(host_stop - host_start).count();
        if (!img.input_host) {
            std::cerr << "Failed to load " << entry.path() << "\n";
            continue;
        }
        img.width = w; img.height = h; img.channels_in = 3;
        img.channels_out = output_channels;
        size_t output_size = (size_t)w * h * img.channels_out;
        img.output_host = new unsigned char[output_size];
        images.push_back(img);
    }
    if (images.empty()) {
        std::cerr << "No images found in " << folder << "\n";
        return 1;
    }

    // --- 5. Process all images (with per-image timing) ---
    for (auto& img : images) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        if (op == "grayscale") {
            apply_grayscale(img.input_host, img.output_host, img.width, img.height, img.channels_in);
        } else if (op == "gaussian") {
            apply_gaussian(img.input_host, img.output_host, img.width, img.height, img.channels_in,
                           GAUSSIAN_27x27, KERNEL_SIZE_GAUSSIAN);
        } else if (op == "sobel") {
            apply_sobel(img.input_host, img.output_host, img.width, img.height, img.channels_in,
                        sobel_x, sobel_y, KERNEL_SIZE_SOBEL);
        }
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        img.time_process_ms = std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count();
    }

    // --- 6. Save all outputs to disk (MULTITHREADED) ---
    std::vector<std::thread> save_threads;
    for (auto& img : images) {
        save_threads.emplace_back([&img, output_folder, op]() {
            std::string outPath = (fs::path(output_folder) / (img.name + "_" + op + ".png")).string();
            int stride = img.width * img.channels_out;
            auto host_start = std::chrono::high_resolution_clock::now();
            stbi_write_png(outPath.c_str(), img.width, img.height,
                           img.channels_out, img.output_host, stride);
            auto host_stop = std::chrono::high_resolution_clock::now();
            img.time_save_ms = std::chrono::duration<float, std::milli>(host_stop - host_start).count();
            stbi_image_free(img.input_host);
            delete[] img.output_host;
            img.input_host = nullptr;
            img.output_host = nullptr;
        });
    }
    for (auto& t : save_threads) { t.join(); }

    // --- 7. Calculate Totals and Print JSON to stdout ---
    float total_load_ms = 0.0f;
    float total_process_ms = 0.0f;
    float total_export_ms = 0.0f;

    for (const auto& img : images) {
        total_load_ms += img.time_load_ms;
        total_process_ms += img.time_process_ms;
        total_export_ms += img.time_save_ms;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n";
    std::cout << "  \"total_loading_time\": " << total_load_ms << ",\n";
    std::cout << "  \"total_processing_time\": " << total_process_ms << ",\n";
    std::cout << "  \"total_exporting_time\": " << total_export_ms << ",\n";
    std::cout << "  \"individual_image_times\": [\n";

    for (size_t i = 0; i < images.size(); ++i) {
        const auto& img = images[i];
        std::cout << "    {\n";
        std::cout << "      \"image_name\": \"" << img.get_json_name() << "\",\n";
        std::cout << "      \"load_ms\": " << img.time_load_ms << ",\n";
        std::cout << "      \"process_ms\": " << img.time_process_ms << ",\n";
        std::cout << "      \"export_ms\": " << img.time_save_ms << "\n";
        std::cout << "    }" << (i == images.size() - 1 ? "" : ",") << "\n";
    }

    std::cout << "  ]\n";
    std::cout << "}\n";

    // Create file path for JSON
    std::string json_path = (fs::path(output_folder) / "timings.json").string();
    std::ofstream json_file(json_path); // <-- Create file stream

    json_file << std::fixed << std::setprecision(4);
    json_file << "{\n";
    json_file << "  \"total_loading_time\": " << total_load_ms << ",\n";
    json_file << "  \"total_processing_time\": " << total_process_ms << ",\n";
    json_file << "  \"total_exporting_time\": " << total_export_ms << ",\n";
    json_file << "  \"individual_image_times\": [\n";

    for (size_t i = 0; i < images.size(); ++i) {
        const auto& img = images[i];
        json_file << "    {\n";
        json_file << "      \"image_name\": \"" << img.get_json_name() << "\",\n";
        json_file << "      \"load_ms\": " << img.time_load_ms << ",\n";
        json_file << "      \"process_ms\": " << img.time_process_ms << ",\n";
        json_file << "      \"export_ms\": " << img.time_save_ms << "\n";
        json_file << "    }" << (i == images.size() - 1 ? "" : ",") << "\n";
    }

    json_file << "  ]\n";
    json_file << "}\n";
    json_file.close(); // <-- Close the file

    return 0;
}