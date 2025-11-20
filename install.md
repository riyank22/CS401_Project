# Installation & Build Guide

This document explains how to compile all C++ compute engines (CUDA, OpenMP, Single-Thread, MPI) and how to build and run the full-stack application.

---

# Project Structure

```
/ProjectRoot/
│
├── cpp_code/
│   ├── GPU.cu
│   ├── OMP.cpp
│   ├── cpu_single.cpp
│   ├── MPI.cpp
│   ├── filters.cuh
│   ├── stb_image.h
│   ├── stb_image_write.h
│   ├── CMakeLists.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│
└── backend/
    ├── bin/
    ├── public/
    ├── index.js
    ├── package.json
```

---

# Part 1: Build the C++ Compute Engines

## Prerequisites
- g++ (C++20)
- CMake 3.20+
- CUDA Toolkit 12.0+ (with `nvcc`)
- OpenMPI 4.1+ (`mpirun`)

## Steps

### 1. Go to the C++ directory
```bash
cd /path/to/ProjectRoot/cpp_code
```

### 2. Configure CMake
```bash
cmake -B build
```

### 3. Build executables
```bash
cmake --build build -j
```

### 4. Create backend/bin folder
```bash
mkdir -p ../backend/bin
```

### 5. Copy binaries to backend
```bash
cp cpp_code/build/GPU ../backend/bin/GPU
cp cpp_code/build/OMP ../backend/bin/OMP
cp cpp_code/build/ProjectCode_ST ../backend/bin/ST
cp cpp_code/build/Proc_MPI ../backend/bin/MPI
```

---

# Part 2: Build the React Frontend

### 1. Enter frontend directory
```bash
cd /path/to/ProjectRoot/frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Build React app
```bash
npm run build
```

### 4. Ensure backend/public exists
```bash
mkdir -p ../backend/public
```

### 5. Move build to backend/public
```bash
mv build/* ../backend/public/
```

### 6. Fix asset paths (optional)
```bash
sed -i 's/href="\//href="\/public\//g' ../backend/public/index.html
sed -i 's/src="\//src="\/public\//g' ../backend/public/index.html
```

---

# Part 3: Run the Node.js Backend

### 1. Enter backend directory
```bash
cd /path/to/ProjectRoot/backend
```

### 2. Install backend dependencies
```bash
npm install
```

### 3. Start server
```bash
node index.js
```

Server runs at:
```
http://localhost:3001
```

---

# Troubleshooting (MPI)

### If mpirun gives `fork` or `execve` errors:
Make sure your backend points to the correct binary:

```js
const mpiPath = './bin/MPI';
```

### If you get oversubscription errors:
Add `--oversubscribe`:

```js
const args = [
  '-n', '4',
  '--oversubscribe',
  mpiPath,
  inputFolder,
  outputFolder,
  operation
];
const child = spawn('mpirun', args);
```

---

# Finished!
All C++ engines compiled, frontend built, backend running.
