//
// Created by riyank on 11/11/25.
//
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <mpi.h> // <-- MPI Header

// STB Image libraries
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

// (Image struct is unchanged)
struct Image {
    std::string name, ext;
    int width, height, channels_in;
    int channels_out;
    unsigned char* input_host;
    unsigned char* output_host;
    float time_load_ms = 0.0f;
    float time_process_ms = 0.0f;
    float time_save_ms = 0.0f;

    // Path for loading
    std::string input_path;

    std::string get_json_name() const { /* ... (same as before) */ }
};

// ==================== KERNEL DEFINITIONS ====================
const float GAUSSIAN_27x27[729] = { /* ... your 729 values ... */ 0.00000102f };
const float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
const float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
// ========================================================

// ==================== CPU FILTER FUNCTIONS (SINGLE-THREADED) ====================
// (These are the same single-threaded functions from cpu_single.cpp)
inline int clamp(int val, int min_val, int max_val) { /* ... */ }
inline float clamp_f(float val, float min_val, float max_val) { /* ... */ }
void apply_grayscale(const unsigned char* in, unsigned char* out, int w, int h, int c_in) { /* ... */ }
void apply_gaussian(const unsigned char* in, unsigned char* out, int w, int h, int c_in,
                    const float* kernel, int k_size) { /* ... */ }
void apply_sobel(const unsigned char* in, unsigned char* out, int w, int h, int c_in,
                 const float* kx_kernel, const float* ky_kernel, int k_size) { /* ... */ }
// ========================================================


int main(int argc, char** argv)
{
    // --- 1. MPI Initialization ---
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // --- 2. Argument and Folder Setup ---
    if (argc < 4) {
        if (world_rank == 0) { // Only Rank 0 prints errors
            std::cerr << "Usage: mpirun -n <processes> ./Proc_MPI <input_folder> <output_folder> <operation>\n";
        }
        MPI_Finalize();
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
    else { /* ... error ... */ }

    if (world_rank == 0) {
        fs::create_directories(output_folder);
    }

    // --- 3. Rank 0: Find files and Broadcast ---
    int num_files = 0;
    std::vector<std::string> file_paths;

    if (world_rank == 0) {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                file_paths.push_back(entry.path().string());
            }
        }
        num_files = file_paths.size();
    }

    // Broadcast the number of files
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0: Send file path lengths, then paths
    if (world_rank == 0) {
        for (const std::string& path : file_paths) {
            int len = path.length();
            MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)path.c_str(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
        }
    } else { // Other ranks: Receive file paths
        file_paths.resize(num_files);
        for (int i = 0; i < num_files; ++i) {
            int len;
            MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
            char* buf = new char[len + 1];
            MPI_Bcast(buf, len, MPI_CHAR, 0, MPI_COMM_WORLD);
            buf[len] = '\0';
            file_paths[i] = std::string(buf);
            delete[] buf;
        }
    }

    if (num_files == 0) {
        if (world_rank == 0) std::cerr << "No images found in " << folder << "\n";
        MPI_Finalize();
        return 1;
    }

    // --- 4. All Ranks: Parallel Load, Process, Save ---
    std::vector<Image> my_images;
    float my_total_load = 0, my_total_process = 0, my_total_save = 0;

    for (int i = world_rank; i < num_files; i += world_size) {
        // This process is responsible for file at index 'i'
        Image img;
        fs::path p(file_paths[i]);
        img.name = p.stem().string();
        img.ext = p.extension().string();

        // --- Load ---
        auto host_start = std::chrono::high_resolution_clock::now();
        int w, h, c;
        img.input_host = stbi_load(file_paths[i].c_str(), &w, &h, &c, 3);
        auto host_stop = std::chrono::high_resolution_clock::now();
        img.time_load_ms = std::chrono::duration<float, std::milli>(host_stop - host_start).count();
        my_total_load += img.time_load_ms;

        if (!img.input_host) {
             std::cerr << "Rank " << world_rank << " failed to load " << file_paths[i] << "\n";
            continue;
        }

        img.width = w; img.height = h; img.channels_in = 3;
        img.channels_out = output_channels;
        size_t output_size = (size_t)w * h * img.channels_out;
        img.output_host = new unsigned char[output_size];

        // --- Process ---
        auto cpu_start = std::chrono::high_resolution_clock::now();
        if (op == "grayscale") { apply_grayscale(img.input_host, img.output_host, w, h, 3); }
        else if (op == "gaussian") { apply_gaussian(img.input_host, img.output_host, w, h, 3, GAUSSIAN_27x27, KERNEL_SIZE_GAUSSIAN); }
        else if (op == "sobel") { apply_sobel(img.input_host, img.output_host, w, h, 3, sobel_x, sobel_y, KERNEL_SIZE_SOBEL); }
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        img.time_process_ms = std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count();
        my_total_process += img.time_process_ms;

        // --- Save ---
        std::string outPath = (fs::path(output_folder) / (img.name + "_" + op + ".png")).string();
        int stride = img.width * img.channels_out;
        auto save_start = std::chrono::high_resolution_clock::now();
        stbi_write_png(outPath.c_str(), img.width, img.height, img.channels_out, img.output_host, stride);
        auto save_stop = std::chrono::high_resolution_clock::now();
        img.time_save_ms = std::chrono::duration<float, std::milli>(save_stop - save_start).count();
        my_total_save += img.time_save_ms;

        // --- Cleanup ---
        stbi_image_free(img.input_host);
        delete[] img.output_host;

        my_images.push_back(img); // Store stats
    }

    // --- 5. Gather All Stats to Rank 0 ---
    float total_load_ms = 0, total_process_ms = 0, total_export_ms = 0;
    MPI_Reduce(&my_total_load, &total_load_ms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_total_process, &total_process_ms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_total_save, &total_export_ms, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0: Gather individual image timings (this is more complex, for now just totals)
    // A full implementation would use MPI_Gatherv to collect all `Image` structs.

    // --- 6. Rank 0: Write JSON File ---
    if (world_rank == 0) {
        std::string json_path = (fs::path(output_folder) / "time.json").string();
        std::ofstream json_file(json_path);

        json_file << std::fixed << std::setprecision(4);
        json_file << "{\n";
        json_file << "  \"total_loading_time\": " << total_load_ms << ",\n";
        json_file << "  \"total_processing_time\": " << total_process_ms << ",\n";
        json_file << "  \"total_exporting_time\": " << total_export_ms << ",\n";

        // Note: This JSON only has totals. Gathering individual timings is
        // a significant effort (sending vectors of structs)
        json_file << "  \"individual_image_times\": [\n";
        json_file << "    {\n";
        json_file << "      \"image_name\": \"MPI_Note\",\n";
        json_file << "      \"load_ms\": 0.0,\n";
        json_file << "      \"process_ms\": 0.0,\n";
        json_file << "      \"export_ms\": 0.0,\n";
        json_file << "      \"note\": \"Individual timings not gathered in this MPI version.\"\n";
        json_file << "    }\n";
        json_file << "  ]\n";
        json_file << "}\n";
        json_file.close();
    }

    // --- 7. Finalize MPI ---
    MPI_Finalize();
    return 0;
}