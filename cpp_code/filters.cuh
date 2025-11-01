#ifndef FILTERS_CUH
#define FILTERS_CUH

#include <cuda_runtime.h>
#include <math.h> // For sqrtf

// ==================== GRAYSCALE ====================
__global__ void grayscale_kernel(const unsigned char* input, unsigned char* output,
                                 int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];
    output[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

// ==================== GAUSSIAN BLUR ====================
// Kernel size is 81 (9x9)
__constant__ float GAUSSIAN_KERNEL[27*27];

__global__ void gaussian_blur_kernel_color(const unsigned char* input,
                                           unsigned char* output,
                                           int width, int height, int channels,
                                           int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = kernel_size / 2;

    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
        for (int kx = -half; kx <= half; ++kx) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int nIdx = (ny * width + nx) * channels; // Input image has 'channels' (3)

            // Corrected kernel index calculation
            float k = GAUSSIAN_KERNEL[(ky + half) * kernel_size + (kx + half)];

            // Ensure we read all 'channels' from the input image
            r_sum += k * input[nIdx];
            g_sum += k * input[nIdx + 1];
            b_sum += k * input[nIdx + 2];
        }
    }

    // Output image has 'channels' (3)
    int outIdx = (y * width + x) * channels;
    output[outIdx]     = static_cast<unsigned char>(min(max(r_sum, 0.0f), 255.0f));
    output[outIdx + 1] = static_cast<unsigned char>(min(max(g_sum, 0.0f), 255.0f));
    output[outIdx + 2] = static_cast<unsigned char>(min(max(b_sum, 0.0f), 255.0f));
}


// ==================== SOBEL EDGE DETECTION ====================
__constant__ float SOBEL_X[9];
__constant__ float SOBEL_Y[9];

__global__ void sobel_filter_kernel(const unsigned char* input, unsigned char* output,
                                    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float gx = 0.0f;
    float gy = 0.0f;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int idx = (ny * width + nx) * channels; // Input image has 'channels' (3)
            // Compute grayscale value internally for Sobel
            float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];

            gx += gray * SOBEL_X[(ky + 1) * 3 + (kx + 1)];
            gy += gray * SOBEL_Y[(ky + 1) * 3 + (kx + 1)];
        }
    }

    float mag = sqrtf(gx * gx + gy * gy);
    output[y * width + x] = static_cast<unsigned char>(min(max(mag, 0.0f), 255.0f));
}

#endif // FILTERS_CUH