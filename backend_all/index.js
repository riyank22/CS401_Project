/*
 * =============================================================================
 * Advanced Benchmark Interface (v1.0) - Backend Server
 * =============================================================================
 *
 * This file contains the decoupled Node.js / Express backend server.
 * It serves a JSON API and manages file-based job processing.
 *
 * == BACKEND (Node.js) ==
 * API:
 * - POST /api/jobs/submit      : Upload images, create job, start background task.
 * - GET  /api/jobs/status/:jobId : Poll for job status (reads status.json).
 * - GET  /api/jobs/result/:jobId : Get full (partial) results for a job.
 * - GET  /api/jobs/previous/:jobId : Alias for /result.
 * - GET  /public/* : Serves static files from the ./public directory.
 *
 * File Structure Managed:
 * ./public/
 * └── jobs/
 * └── <job_id>/
 * ├── input/
 * ├── output_cuda/
 * ├── output_openmp/
 * ├── output_mpi/
 * └── status.json
 *
 * Binaries (Assumed Location):
 * ./bin/
 * ├── GPU
 * ├── OMP
 * └── GPU (for MPI, as per request)
 *
 * == HOW TO RUN ==
 * 1. Install dependencies:
 * npm install express multer fs-extra uuid image-size cors
 * 2. Create a `./bin` directory and place your three executables inside.
 * 3. Create a `./public` directory.
 * 4. Run the server:
 * node backend_server.js
 * 5. Test the API using a tool like Postman or curl.
 *
 */

// =============================================================================
//   1. BACKEND: IMPORTS & SETUP
// =============================================================================
const express = require('express');
const multer = require('multer');
const fs = require('fs-extra');
const path = require('path');
const { execFile } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const sizeOf = require('image-size');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3001;
const upload = multer({ dest: 'uploads/' }); // Temp storage for uploads

// --- File Path Configuration ---
const publicDir = path.join(__dirname, 'public');
const jobsDir = path.join(publicDir, 'jobs');
const binDir = path.join(__dirname, 'bin');

// --- Middleware ---
app.use(cors()); // Allow cross-origin requests
app.use(express.json()); // Parse JSON bodies
app.use('/public', express.static(publicDir)); // Serve static files

// Ensure required directories exist
fs.ensureDirSync(jobsDir);
fs.ensureDirSync(binDir);

// =============================================================================
//   2. BACKEND: HELPER FUNCTIONS
// =============================================================================

/**
 * Safely reads and updates the status.json file for a job.
 * @param {string} jobId - The ID of the job.
 * @param {object} updates - An object with keys/values to merge into the status.
 */
async function updateStatus(jobId, updates) {
  const statusPath = path.join(jobsDir, jobId, 'status.json');
  try {
    const status = await fs.readJson(statusPath);
    const newStatus = { ...status, ...updates, last_updated: new Date().toISOString() };
    await fs.writeJson(statusPath, newStatus, { spaces: 2 });
    return newStatus;
  } catch (error) {
    console.error(`[Job ${jobId}] Failed to update status:`, error);
  }
}

/**
 * Runs a C++ executable and returns a promise.
 * @param {string} exeName - Name of the binary (e.g., 'filter_cuda').
 * @param {string} inputDir - Full path to the input directory.
 * @param {string} outputDir - Full path to the output directory.
 * @param {string} filterType - The filter name (e.g., 'grayscale').
 */
function runExecutable(jobId, exeName, inputDir, outputDir, filterType) {
  const exePath = path.join(binDir, exeName);
  const args = [inputDir, outputDir, filterType];

  console.log(`[Job ${jobId}] Running: ${exePath} ${args.join(' ')}`);

  return new Promise((resolve, reject) => {
    execFile(exePath, args, (error, stdout, stderr) => {
      if (error) {
        console.error(`[Job ${jobId}] Executable Error (${exeName}):`, stderr);
        return reject(new Error(`Error running ${exeName}: ${stderr || error.message}`));
      }
      console.log(`[Job ${jobId}] Executable Output (${exeName}):`, stdout);
      resolve(stdout);
    });
  });
}

/**
 * The main asynchronous background task for running the sequential benchmark.
 * This is "fire-and-forget" from the /submit endpoint.
 * @param {string} jobId - The ID of the job to process.
 * @param {string} filterType - The filter name.
 */
async function runFullBenchmarkSequentially(jobId, filterType) {
  const jobDir = path.join(jobsDir, jobId);
  const inputDir = path.join(jobDir, 'input');
  let currentStep = '';

  try {
    // 1. CUDA
    currentStep = 'cuda';
    const cudaOutDir = path.join(jobDir, 'output_cuda');
    await updateStatus(jobId, { status: 'processing_cuda', cuda: 'processing' });
    await runExecutable(jobId, 'GPU', inputDir, cudaOutDir, filterType); // Changed from 'filter_cuda'
    await updateStatus(jobId, { cuda: 'completed' });

    // 2. OpenMP
    currentStep = 'openmp';
    const openmpOutDir = path.join(jobDir, 'output_openmp');
    await updateStatus(jobId, { status: 'processing_openmp', openmp: 'processing' });
    await runExecutable(jobId, 'OMP', inputDir, openmpOutDir, filterType); // Changed from 'filter_openmp'
    await updateStatus(jobId, { openmp: 'completed' });

    // 3. MPI (using GPU executable as requested)
    currentStep = 'mpi';
    const mpiOutDir = path.join(jobDir, 'output_mpi');
    await updateStatus(jobId, { status: 'processing_mpi', mpi: 'processing' });
    await runExecutable(jobId, 'GPU', inputDir, mpiOutDir, filterType); // Changed from 'filter_mpi'
    await updateStatus(jobId, { mpi: 'completed' });

    // 4. All Done
    await updateStatus(jobId, { status: 'completed' });
    console.log(`[Job ${jobId}] Benchmark completed successfully.`);

  } catch (error) {
    console.error(`[Job ${jobId}] Benchmark FAILED at step '${currentStep}':`, error);
    // Mark the failing step and stop the job
    await updateStatus(jobId, {
      status: 'failed',
      [currentStep]: 'failed',
      error: error.message
    });
  }
}

/**
 * Helper to read an engine's result data if it's completed.
 * @param {string} engineName - 'cuda', 'openmp', or 'mpi'.
 * @param {object} status - The job's status object.
 * @param {string} jobDir - Full path to the job directory.
 * @param {number} batchSize - Number of images, for calculating averages.
 * @returns {object | null} - The data object or null.
 */
async function getEngineData(engineName, status, jobDir, batchSize) {
  if (status[engineName] !== 'completed') {
    return null;
  }
  try {
    const timingsPath = path.join(jobDir, `output_${engineName}`, 'timings.json');
    const timings = await fs.readJson(timingsPath);
    // Tweak 2: Return the timings object directly without creating a summary.
    return timings;
  } catch (error) {
    console.error(`Could not read timings for ${engineName}:`, error);
    return { error: 'Could not read timings.json' };
  }
}

/**
 * The main "smart" handler for GET /result and GET /previous.
 * Gathers all available data for a job and builds the response.
 */
async function getJobResultHandler(req, res) {
  const { job_id } = req.params;
  const jobDir = path.join(jobsDir, job_id);

  try {
    // 1. Get Status
    const statusPath = path.join(jobDir, 'status.json');
    if (!await fs.pathExists(statusPath)) {
      return res.status(404).json({ error: 'Job not found' });
    }
    const status = await fs.readJson(statusPath);

    // 2. Get Input Images
    const inputDir = path.join(jobDir, 'input');
    let inputImages = [];
    let batchSize = 0;

    const operation = status.filter_type;

    try {
      const filenames = await fs.readdir(inputDir);
      for (const filename of filenames) {
        const imgPath = path.join(inputDir, filename);
        // const dimensions = sizeOf(imgPath);
        
        // Tweak 4: Add conditional output URLs
        const imageObject = {
          filename: filename,
          url: `/public/jobs/${job_id}/input/${filename}`,
        //   width: dimensions.width,
        //   height: dimensions.height,
        };

        const parsed = path.parse(filename);
        const baseName = parsed.name;
        const outFilename = `${baseName}_${operation}.png`;

        if (status.cuda === 'completed') {
          imageObject.cuda_output_url = `/public/jobs/${job_id}/output_cuda/${outFilename}`;
        }
        if (status.openmp === 'completed') {
          imageObject.openmp_output_url = `/public/jobs/${job_id}/output_openmp/${outFilename}`;
        }
        if (status.mpi === 'completed') {
          imageObject.mpi_output_url = `/public/jobs/${job_id}/output_mpi/${outFilename}`;
        }

        inputImages.push(imageObject);
      }
      batchSize = inputImages.length;
    } catch (e) {
      console.warn(`[Job ${job_id}] Could not read input images:`, e.message);
    }

    // 3. Conditionally get results for each engine
    const [cudaData, openmpData, mpiData] = await Promise.all([
      getEngineData('cuda', status, jobDir, batchSize),
      getEngineData('openmp', status, jobDir, batchSize),
      getEngineData('mpi', status, jobDir, batchSize)
    ]);

    // 4. Assemble final response
    const response = {
      job_id: job_id,
      status_details: status,
      input_images: inputImages,
      batch_size: batchSize,
      cuda_data: cudaData,
      openmp_data: openmpData,
      mpi_data: mpiData,
    };

    res.json(response);
  } catch (error) {
    console.error(`[Job ${job_id}] Error in getJobResultHandler:`, error);
    res.status(500).json({ error: 'Failed to retrieve job results' });
  }
}

async function getJobResultList(req, res)
{
  try {
    const entries = await fs.readdir(jobsDir);
    const jobIds = [];
    for (const name of entries) {
      const fullPath = path.join(jobsDir, name);
      try {
        const stat = await fs.stat(fullPath);
        if (stat.isDirectory() && name.startsWith('job_')) jobIds.push(name);
      } catch (e) {
        // ignore unreadable entries
      }
    }
    res.json({ jobs: jobIds });
  } catch (error) {
    console.error('Failed to list jobs:', error);
    res.status(500).json({ error: 'Failed to list jobs' });
  }
}

// =============================================================================
//   3. BACKEND: API ENDPOINTS
// =============================================================================

/**
 * ENDPOINT 1: Submit Job
 * - Accepts multipart/form-data with 'files' and 'filter_type'.
 * - Creates job structure.
 * - Starts background processing.
 * - Responds 202 Accepted.
 */
app.post('/api/jobs/submit', upload.array('files'), async (req, res) => {
  const { filter_type } = req.body;
  const files = req.files;

  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'No files uploaded.' });
  }
  if (!filter_type) {
    return res.status(400).json({ error: 'No filter_type specified.' });
  }

  const jobId = `job_${uuidv4()}`;
  const jobDir = path.join(jobsDir, jobId);
  const inputDir = path.join(jobDir, 'input');

  try {
    // 1. Create directory structure
    await fs.ensureDir(inputDir);
    await fs.ensureDir(path.join(jobDir, 'output_cuda'));
    await fs.ensureDir(path.join(jobDir, 'output_openmp'));
    await fs.ensureDir(path.join(jobDir, 'output_mpi'));

    // 2. Move uploaded files to input directory
    for (const file of files) {
      await fs.move(file.path, path.join(inputDir, file.originalname));
    }

    // 3. Create initial status.json
    const initialState = {
      job_id: jobId,
      filter_type: filter_type,
      status: 'pending',
      cuda: 'pending',
      openmp: 'pending',
      mpi: 'pending',
      submitted_at: new Date().toISOString(),
      batch_size: files.length,
    };
    await fs.writeJson(path.join(jobDir, 'status.json'), initialState, { spaces: 2 });

    // 4. Start background task (DO NOT await)
    runFullBenchmarkSequentially(jobId, filter_type);

    // 5. Respond 202 Accepted
    res.status(202).json(initialState);

  } catch (error) {
    console.error(`[Job ${jobId}] Failed to submit job:`, error);
    res.status(500).json({ error: 'Job submission failed.' });
    // Cleanup failed job directory
    await fs.remove(jobDir);
  }
});

/**
 * ENDPOINT 2: Get Job Status (Polling)
 * - Reads and returns the status.json file.
 */
app.get('/api/jobs/status/:job_id', async (req, res) => {
  const { job_id } = req.params;
  const statusPath = path.join(jobsDir, job_id, 'status.json');

  try {
    const status = await fs.readJson(statusPath);
    res.json(status);
  } catch (error) {
    res.status(404).json({ error: 'Job not found or status file unreadable.' });
  }
});

/**
 * ENDPOINT 3: Get Job Results (Main Data)
 * - Calls the "smart" handler.
 */
app.get('/api/jobs/result/:job_id', getJobResultHandler);

/**
 * ENDPOINT 4: Get Previous Job (Alias)
 * - Calls the "smart" handler.
 */
app.get('/api/jobs/list', getJobResultList);

// =============================================================================
//   4. START THE SERVER
// =============================================================================

// Serve the SPA entrypoint
app.get('/', (req, res) => {
    const indexPath = path.join(publicDir, 'index.html');
    if (!fs.existsSync(indexPath)) {
        return res.status(404).send('index.html not found');
    }
    res.sendFile(indexPath);
});

app.listen(port, () => {
  console.log(`=======================================================`);
  console.log(`  Benchmark API Server is running!`);
  console.log(`  Backend API: http://localhost:${port}/api/...`);
  console.log(`  Static Files: http://localhost:${port}/public/...`);
  console.log(`=======================================================`);
  console.log(`\nMonitoring for jobs...`);
});

