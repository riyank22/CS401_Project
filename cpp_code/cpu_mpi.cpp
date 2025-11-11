// filter_mpi_all.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

struct ImageTiming {
    std::string image_name;
    double load_ms;
    double process_ms;
    double export_ms;
};

// Utility ---------------------------------------------------------
inline unsigned char to_gray_uc(unsigned char r, unsigned char g, unsigned char b) {
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    int gi = static_cast<int>(gray + 0.5f);
    if (gi < 0) gi = 0; if (gi > 255) gi = 255;
    return static_cast<unsigned char>(gi);
}

inline unsigned char clamp_uc(float v) {
    int iv = static_cast<int>(v + 0.5f);
    if (iv < 0) iv = 0;
    if (iv > 255) iv = 255;
    return static_cast<unsigned char>(iv);
}

// ---------------------------------------------------------------
// Perform grayscale (no halo exchange)
void mpi_grayscale(const std::string &input_path, const std::string &output_path,
                   int rank, int size,
                   std::vector<std::tuple<std::string,double,double,double>> &timings)
{
    int width=0, height=0, channels=3;
    unsigned char *full_img=nullptr;
    double t0 = MPI_Wtime();

    if(rank==0){
        full_img = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
        if(!full_img){
            std::cerr<<"Failed to load "<<input_path<<"\n";
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        std::cout<<"Loaded "<<input_path<<" ("<<width<<"x"<<height<<")\n";
    }

    MPI_Bcast(&width,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&height,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&channels,1,MPI_INT,0,MPI_COMM_WORLD);

    int base=height/size, rem=height%size;
    int myrows = base + (rank<rem?1:0);

    std::vector<int> sendcounts(size), displs(size);
    if(rank==0){
        int off=0;
        for(int r=0;r<size;r++){
            int rows = base + (r<rem?1:0);
            sendcounts[r]=rows*width*channels;
            displs[r]=off;
            off+=sendcounts[r];
        }
    }

    std::vector<unsigned char> local_rgb(myrows*width*channels);
    MPI_Scatterv(full_img,sendcounts.data(),displs.data(),MPI_UNSIGNED_CHAR,
                 local_rgb.data(),local_rgb.size(),MPI_UNSIGNED_CHAR,
                 0,MPI_COMM_WORLD);

    if(rank==0){ stbi_image_free(full_img); }

    double t1 = MPI_Wtime();

    std::vector<unsigned char> local_gray(myrows*width);
    for(int y=0;y<myrows;y++)
        for(int x=0;x<width;x++){
            int idx=(y*width+x)*channels;
            local_gray[y*width+x]=to_gray_uc(local_rgb[idx],local_rgb[idx+1],local_rgb[idx+2]);
        }

    double t2 = MPI_Wtime();

    std::vector<int> recvcounts(size), displs2(size);
    if(rank==0){
        int off=0;
        for(int r=0;r<size;r++){
            int rows = base + (r<rem?1:0);
            recvcounts[r]=rows*width;
            displs2[r]=off;
            off+=recvcounts[r];
        }
    }
    std::vector<unsigned char> full_gray;
    if(rank==0) full_gray.resize(width*height);

    MPI_Gatherv(local_gray.data(),local_gray.size(),MPI_UNSIGNED_CHAR,
                full_gray.data(),recvcounts.data(),displs2.data(),
                MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

    double t3 = MPI_Wtime();

    if(rank==0){
        stbi_write_png(output_path.c_str(),width,height,1,full_gray.data(),width);
        timings.push_back({"grayscale",(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000});
        std::cout<<"[Grayscale] saved "<<output_path<<"\n";
    }
}

// ---------------------------------------------------------------
// Perform Sobel edge detection (with halo exchange)
void mpi_sobel(const std::string &input_path, const std::string &output_path,
               int rank, int size,
               std::vector<std::tuple<std::string,double,double,double>> &timings)
{
    int width=0, height=0, channels=3;
    unsigned char *full_img=nullptr;
    double t0 = MPI_Wtime();

    if(rank==0){
        full_img = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
        if(!full_img){ std::cerr<<"Failed to load "<<input_path<<"\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    }

    MPI_Bcast(&width,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&height,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&channels,1,MPI_INT,0,MPI_COMM_WORLD);

    int base=height/size, rem=height%size;
    int myrows = base + (rank<rem?1:0);

    std::vector<int> sendcounts(size), displs(size);
    if(rank==0){
        int off=0;
        for(int r=0;r<size;r++){
            int rows = base + (r<rem?1:0);
            sendcounts[r]=rows*width*channels;
            displs[r]=off;
            off+=sendcounts[r];
        }
    }

    std::vector<unsigned char> local_rgb(myrows*width*channels);
    MPI_Scatterv(full_img,sendcounts.data(),displs.data(),MPI_UNSIGNED_CHAR,
                 local_rgb.data(),local_rgb.size(),MPI_UNSIGNED_CHAR,
                 0,MPI_COMM_WORLD);
    if(rank==0) stbi_image_free(full_img);

    double t1 = MPI_Wtime();

    // Convert to grayscale
    std::vector<unsigned char> local_gray(myrows*width);
    for(int y=0;y<myrows;y++)
        for(int x=0;x<width;x++){
            int idx=(y*width+x)*channels;
            local_gray[y*width+x]=to_gray_uc(local_rgb[idx],local_rgb[idx+1],local_rgb[idx+2]);
        }

    // Halo exchange (1 row up/down)
    std::vector<unsigned char> top(width), bottom(width);
    int above=(rank==0)?MPI_PROC_NULL:rank-1;
    int below=(rank==size-1)?MPI_PROC_NULL:rank+1;

    MPI_Sendrecv(local_gray.data(),width,MPI_UNSIGNED_CHAR,above,0,
                 top.data(),width,MPI_UNSIGNED_CHAR,above,1,
                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Sendrecv(local_gray.data()+(myrows-1)*width,width,MPI_UNSIGNED_CHAR,below,1,
                 bottom.data(),width,MPI_UNSIGNED_CHAR,below,0,
                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    // Compute Sobel
    std::vector<unsigned char> local_edge(myrows*width);
    const int kx[9]={-1,0,1,-2,0,2,-1,0,1};
    const int ky[9]={-1,-2,-1,0,0,0,1,2,1};

    auto get_gray=[&](int y,int x)->unsigned char{
        if(y<0) return top[x];
        if(y>=myrows) return bottom[x];
        if(x<0) x=0; if(x>=width) x=width-1;
        return local_gray[y*width+x];
    };

    for(int y=0;y<myrows;y++){
        for(int x=0;x<width;x++){
            float gx=0,gy=0; int idx=0;
            for(int ky_= -1;ky_<=1;ky_++)
                for(int kx_= -1;kx_<=1;kx_++){
                    unsigned char v=get_gray(y+ky_,x+kx_);
                    gx+=v*kx[idx]; gy+=v*ky[idx]; idx++;
                }
            local_edge[y*width+x]=clamp_uc(std::sqrt(gx*gx+gy*gy));
        }
    }

    double t2 = MPI_Wtime();

    std::vector<int> recvcounts(size), displs2(size);
    if(rank==0){
        int off=0;
        for(int r=0;r<size;r++){
            int rows = base + (r<rem?1:0);
            recvcounts[r]=rows*width;
            displs2[r]=off;
            off+=recvcounts[r];
        }
    }

    std::vector<unsigned char> full_edge;
    if(rank==0) full_edge.resize(width*height);

    MPI_Gatherv(local_edge.data(),local_edge.size(),MPI_UNSIGNED_CHAR,
                full_edge.data(),recvcounts.data(),displs2.data(),
                MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

    double t3 = MPI_Wtime();

    if(rank==0){
        stbi_write_png(output_path.c_str(),width,height,1,full_edge.data(),width);
        timings.push_back({"sobel",(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000});
        std::cout<<"[Sobel] saved "<<output_path<<"\n";
    }
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// Perform Gaussian Blur (9x9) on RGB channels with halo exchange (radius = 4)
void mpi_gaussian(const std::string &input_path, const std::string &output_path,
                  int rank, int size,
                  std::vector<std::tuple<std::string,double,double,double>> &timings)
{
    const int R = 4;  // halo radius for 9x9 filter
    const int K = 9;  // kernel size
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

    int width=0, height=0, channels=3;
    unsigned char *full_img=nullptr;
    double t0 = MPI_Wtime();

    if(rank==0){
        full_img = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
        if(!full_img){ std::cerr<<"Failed to load "<<input_path<<"\n"; MPI_Abort(MPI_COMM_WORLD,1); }
        channels = 3;
    }

    MPI_Bcast(&width,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&height,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&channels,1,MPI_INT,0,MPI_COMM_WORLD);

    int base=height/size, rem=height%size;
    int myrows = base + (rank<rem?1:0);

    std::vector<int> sendcounts(size), displs(size);
    if(rank==0){
        int off=0;
        for(int r=0;r<size;r++){
            int rows = base + (r<rem?1:0);
            sendcounts[r]=rows*width*channels;
            displs[r]=off;
            off+=sendcounts[r];
        }
    }

    std::vector<unsigned char> local_rgb(myrows*width*channels);
    MPI_Scatterv(full_img,sendcounts.data(),displs.data(),MPI_UNSIGNED_CHAR,
                 local_rgb.data(),local_rgb.size(),MPI_UNSIGNED_CHAR,
                 0,MPI_COMM_WORLD);
    if(rank==0) stbi_image_free(full_img);

    double t1 = MPI_Wtime();

    // Halo exchange (4 rows * width * 3 channels)
    std::vector<unsigned char> top(width*channels*R), bottom(width*channels*R);
    int above=(rank==0)?MPI_PROC_NULL:rank-1;
    int below=(rank==size-1)?MPI_PROC_NULL:rank+1;

    MPI_Sendrecv(local_rgb.data(),width*channels*R,MPI_UNSIGNED_CHAR,above,0,
                 top.data(),width*channels*R,MPI_UNSIGNED_CHAR,above,1,
                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Sendrecv(local_rgb.data()+(myrows-R)*width*channels,width*channels*R,MPI_UNSIGNED_CHAR,below,1,
                 bottom.data(),width*channels*R,MPI_UNSIGNED_CHAR,below,0,
                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    // Apply Gaussian 9x9 per channel
    std::vector<unsigned char> local_blur(myrows*width*channels);

    auto get_rgb=[&](int y,int x,int c)->unsigned char{
        if(y<0) return top[(R+y)*width*channels + x*channels + c];
        if(y>=myrows) return bottom[(y-myrows)*width*channels + x*channels + c];
        if(x<0) x=0; if(x>=width) x=width-1;
        return local_rgb[y*width*channels + x*channels + c];
    };

    for(int y=0;y<myrows;y++){
        for(int x=0;x<width;x++){
            float acc[3]={0,0,0};
            for(int ky=-R;ky<=R;ky++){
                for(int kx=-R;kx<=R;kx++){
                    float w = GAUSSIAN_9x9[(ky+R)*K + (kx+R)];
                    for(int c=0;c<3;c++)
                        acc[c] += get_rgb(y+ky,x+kx,c)*w;
                }
            }
            for(int c=0;c<3;c++)
                local_blur[y*width*channels + x*channels + c] = clamp_uc(acc[c]);
        }
    }

    double t2 = MPI_Wtime();

    // Gather all RGB parts
    std::vector<int> recvcounts(size), displs2(size);
    if(rank==0){
        int off=0;
        for(int r=0;r<size;r++){
            int rows = base + (r<rem?1:0);
            recvcounts[r]=rows*width*channels;
            displs2[r]=off;
            off+=recvcounts[r];
        }
    }

    std::vector<unsigned char> full_blur;
    if(rank==0) full_blur.resize(width*height*channels);

    MPI_Gatherv(local_blur.data(),local_blur.size(),MPI_UNSIGNED_CHAR,
                full_blur.data(),recvcounts.data(),displs2.data(),
                MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

    double t3 = MPI_Wtime();

    if(rank==0){
        stbi_write_png(output_path.c_str(),width,height,channels,full_blur.data(),width*channels);
        timings.push_back({"gaussian_9x9",(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000});
        std::cout<<"[Gaussian 9x9 RGB] saved "<<output_path<<"\n";
    }
}

// ------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            std::cerr << "Usage: mpirun -np <N> ./filter_mpi_all <input_dir> <output_dir> <operation>\n"
                      << "Operation: grayscale | gaussian | sobel\n";
        MPI_Finalize();
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    std::string operation = argv[3];

    if (rank == 0)
        std::cout << "Performing '" << operation << "' on images in " << input_dir
                  << " using " << size << " MPI ranks.\n";

    // Collect input images
    std::vector<std::string> images;
    if (fs::is_directory(input_dir)) {
        for (auto &entry : fs::directory_iterator(input_dir)) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
                images.push_back(entry.path().string());
        }
    } else {
        images.push_back(input_dir);
    }

    std::sort(images.begin(), images.end());

    if (rank == 0)
        std::cout << "Found " << images.size() << " image(s).\n";

    std::vector<ImageTiming> timings;
    double total_load = 0.0, total_process = 0.0, total_export = 0.0;

    for (auto &infile : images) {
        std::string fname = fs::path(infile).filename().string();
        std::string outpath = output_dir + "/" + fs::path(infile).stem().string();

        double t0 = MPI_Wtime();

        // Each operation calls its own MPI routine and returns times
        std::vector<std::tuple<std::string,double,double,double>> op_timings;
        if (operation == "grayscale")
            mpi_grayscale(infile, outpath + "_gray.png", rank, size, op_timings);
        else if (operation == "gaussian")
            mpi_gaussian(infile, outpath + "_gaussian.png", rank, size, op_timings);
        else if (operation == "sobel")
            mpi_sobel(infile, outpath + "_sobel.png", rank, size, op_timings);
        else {
            if (rank == 0)
                std::cerr << "Unknown operation: " << operation << "\n";
            MPI_Finalize();
            return 1;
        }

        // Only rank 0 has timing data
        if (rank == 0 && !op_timings.empty()) {
            auto [name, load, proc, save] = op_timings.back();
            timings.push_back({fname, load, proc, save});
            total_load += load;
            total_process += proc;
            total_export += save;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ---------------- Write JSON Output -----------------
    if (rank == 0) {
        std::ofstream jf(output_dir + "/timings.json");
        jf << std::fixed << std::setprecision(4);
        jf << "{\n";
        jf << "  \"total_loading_time\": " << total_load << ",\n";
        jf << "  \"total_processing_time\": " << total_process << ",\n";
        jf << "  \"total_exporting_time\": " << total_export << ",\n";
        jf << "  \"individual_image_times\": [\n";

        for (size_t i = 0; i < timings.size(); ++i) {
            auto &t = timings[i];
            jf << "    {\n";
            jf << "      \"image_name\": \"" << t.image_name << "\",\n";
            jf << "      \"load_ms\": " << t.load_ms << ",\n";
            jf << "      \"process_ms\": " << t.process_ms << ",\n";
            jf << "      \"export_ms\": " << t.export_ms << "\n";
            jf << "    }" << (i + 1 < timings.size() ? "," : "") << "\n";
        }
        jf << "  ]\n";
        jf << "}\n";
        jf.close();

        std::cout << "Timing data written to " << output_dir << "/timing.json\n";
    }

    MPI_Finalize();
    return 0;
}