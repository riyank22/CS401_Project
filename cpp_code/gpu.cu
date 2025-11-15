#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <string>
#include <numeric>
#include <chrono>
#include <sstream>
#include <cuda_runtime.h>
#include <iomanip> 
#include <fstream>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "filters.cuh"

namespace fs = std::filesystem;

struct Image {
    std::string name, ext;
    int width, height, channels_in;
    int channels_out;
    unsigned char* input_host;
    unsigned char* output_host;

    float time_load_ms = 0.0f;
    float time_gpu_ms = 0.0f;
    float time_save_ms = 0.0f;


    std::string get_json_name() const {
        std::string full_name = name + ext;
        std::string safe_name;
        for (char c : full_name) {
            if (c == '\\') {
                safe_name += "\\\\";
            } else if (c == '"') {
                safe_name += "\\\"";
            } else {
                safe_name += c;
            }
        }
        return safe_name;
    }
};


const float GAUSSIAN_9x9[81] = {
    1.16788635e-02f, 1.19232554e-02f, 1.21009460e-02f, 1.22088289e-02f, 1.22450031e-02f, 1.22088289e-02f, 1.21009460e-02f, 1.19232554e-02f, 1.16788635e-02f,
    1.19232554e-02f, 1.21727614e-02f, 1.23541704e-02f, 1.24643108e-02f, 1.25012421e-02f, 1.24643108e-02f, 1.23541704e-02f, 1.21727614e-02f, 1.19232554e-02f,
    1.21009460e-02f, 1.23541704e-02f, 1.25382828e-02f, 1.26500647e-02f, 1.26875463e-02f, 1.26500647e-02f, 1.25382828e-02f, 1.23541704e-02f, 1.21009460e-02f,
    1.22088289e-02f, 1.24643108e-02f, 1.26500647e-02f, 1.27628431e-02f, 1.28006589e-02f, 1.27628431e-02f, 1.26500647e-02f, 1.24643108e-02f, 1.22088289e-02f,
    1.22450031e-02f, 1.25012421e-02f, 1.26875463e-02f, 1.28006589e-02f, 1.28385867e-02f, 1.28006589e-02f, 1.26875463e-02f, 1.25012421e-02f, 1.22450031e-02f,
    1.22088289e-02f, 1.24643108e-02f, 1.26500647e-02f, 1.27628431e-02f, 1.28006589e-02f, 1.27628431e-02f, 1.26500647e-02f, 1.24643108e-02f, 1.22088289e-02f,
    1.21009460e-02f, 1.23541704e-02f, 1.25382828e-02f, 1.26500647e-02f, 1.26875463e-02f, 1.26500647e-02f, 1.25382828e-02f, 1.23541704e-02f, 1.21009460e-02f,
    1.19232554e-02f, 1.21727614e-02f, 1.23541704e-02f, 1.24643108e-02f, 1.25012421e-02f, 1.24643108e-02f, 1.23541704e-02f, 1.21727614e-02f, 1.19232554e-02f,
    1.16788635e-02f, 1.19232554e-02f, 1.21009460e-02f, 1.22088289e-02f, 1.22450031e-02f, 1.22088289e-02f, 1.21009460e-02f, 1.19232554e-02f, 1.16788635e-02f
};

const float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
const float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};



int main(int argc, char** argv)
{

    if (argc < 4) {
        std::cerr << "Usage: ./ProjectCode <input_folder> <output_folder> <operation>\n";
        std::cerr << "Operations: grayscale | gaussian | sobel\n";
        return 1;
    }

    std::string folder = argv[1];
    std::string output_folder = argv[2];
    std::string op = argv[3];

    const int KERNEL_SIZE_GAUSSIAN = 9;

    int output_channels;
    if (op == "grayscale" || op == "sobel") {
        output_channels = 1;
    } else if (op == "gaussian") {
        output_channels = 3;
    } else {
        std::cerr << "Unknown operation: " << op << "\n";
        return 1;
    }

    fs::create_directories(output_folder);

    std::vector<Image> images;
    size_t max_input_bytes = 0;
    size_t max_output_bytes = 0;

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
        max_input_bytes = std::max(max_input_bytes, (size_t)w * h * img.channels_in);
        max_output_bytes = std::max(max_output_bytes, output_size);
    }
    if (images.empty()) {
        std::cerr << "No images found in " << folder << "\n";
        return 1;
    }

    if (op == "gaussian") {
        cudaMemcpyToSymbol(GAUSSIAN_KERNEL, GAUSSIAN_9x9, sizeof(GAUSSIAN_9x9));
    }
    if (op == "sobel") {
        cudaMemcpyToSymbol(SOBEL_X, sobel_x, sizeof(sobel_x));
        cudaMemcpyToSymbol(SOBEL_Y, sobel_y, sizeof(sobel_y));
    }

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, max_input_bytes);
    cudaMalloc(&d_output, max_output_bytes);
    dim3 block(16, 16);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto& img : images) {
        size_t inSize = (size_t)img.width * img.height * img.channels_in;
        size_t outSize = (size_t)img.width * img.height * img.channels_out;
        cudaEventRecord(start);
        cudaMemcpy(d_input, img.input_host, inSize, cudaMemcpyHostToDevice);
        dim3 grid((img.width + block.x - 1) / block.x, (img.height + block.y - 1) / block.y);
        if (op == "grayscale") {
            grayscale_kernel<<<grid, block>>>(d_input, d_output, img.width, img.height, img.channels_in);
        } else if (op == "gaussian") {
            gaussian_blur_kernel_color<<<grid, block>>>(d_input, d_output, img.width, img.height, img.channels_in, KERNEL_SIZE_GAUSSIAN);
        } else if (op == "sobel") {
            sobel_filter_kernel<<<grid, block>>>(d_input, d_output, img.width, img.height, img.channels_in);
        }
        cudaMemcpy(img.output_host, d_output, outSize, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&img.time_gpu_ms, start, stop);
    }

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

    float total_load_ms = 0.0f;
    float total_process_ms = 0.0f;
    float total_export_ms = 0.0f;

    for (const auto& img : images) {
        total_load_ms += img.time_load_ms;
        total_process_ms += img.time_gpu_ms;
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
        std::cout << "      \"process_ms\": " << img.time_gpu_ms << ",\n";
        std::cout << "      \"export_ms\": " << img.time_save_ms << "\n";
        std::cout << "    }" << (i == images.size() - 1 ? "" : ",") << "\n";
    }

    std::cout << "  ]\n";
    std::cout << "}\n";
    std::string json_path = (fs::path(output_folder) / "timings.json").string();
    std::ofstream json_file(json_path); 

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
        json_file << "      \"process_ms\": " << img.time_gpu_ms << ",\n";
        json_file << "      \"export_ms\": " << img.time_save_ms << "\n";
        json_file << "    }" << (i == images.size() - 1 ? "" : ",") << "\n";
    }

    json_file << "  ]\n";
    json_file << "}\n";
    json_file.close(); 

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}