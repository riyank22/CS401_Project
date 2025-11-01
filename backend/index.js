const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8000;

// --- Configuration ---
const JOBS_DIR = path.join(__dirname, 'jobs');
const INPUT_DIR = path.join(__dirname, 'input');
const EXE_DIR = path.join(__dirname, 'bins');

// --- Middleware ---
app.use(cors()); // Allow cross-origin requests (from our frontend)
app.use(express.json()); // Parse JSON request bodies

// --- Helper Function: Run a benchmark process ---
/**
 * Spawns a new process and returns a Promise that resolves or rejects
 * when the process exits.
 * @param {string} exePath - Path to the executable.
 * @param {string[]} args - Arguments to pass to the executable.
 * @returns {Promise<void>}
 */
function runBenchmarkProcess(jobId, exePath, args) {
  return new Promise((resolve, reject) => {
    console.log(`[${jobId}] Spawning: ${exePath} ${args.join(' ')}`);
    const process = spawn(exePath, args);

    // Log process output for debugging
    process.stdout.on('data', (data) => {
      console.log(`[${jobId}-${path.basename(exePath)}] stdout: ${data.toString().trim()}`);
    });
    process.stderr.on('data', (data) => {
      console.error(`[${jobId}-${path.basename(exePath)}] stderr: ${data.toString().trim()}`);
    });

    // Handle process exit
    process.on('exit', (code) => {
      if (code === 0) {
        console.log(`[${jobId}-${path.basename(exePath)}] Process exited successfully.`);
        resolve();
      } else {
        console.error(`[${jobId}-${path.basename(exePath)}] Process exited with code ${code}.`);
        reject(new Error(`Process ${path.basename(exePath)} exited with code ${code}`));
      }
    });

    // Handle spawn errors
    process.on('error', (err) => {
      console.error(`[${jobId}-${path.basename(exePath)}] Failed to start process:`, err);
      reject(err);
    });
  });
}

// --- Background Task: Run all benchmarks ---
/**
 * This function runs in the background and is NOT awaited by the
 * /start endpoint.
 * @param {string} jobId - The unique ID for this job.
 * @param {string} filterType - The filter type from the user request.
 */
async function runAllBenchmarks(jobId, filterType) {
  const jobDir = path.join(JOBS_DIR, jobId);
  const statusFile = path.join(jobDir, '_STATUS.json');

  try {
    // 1. Update status to "processing"
    await fs.writeFile(statusFile, JSON.stringify({ status: 'processing' }));

    // 2. Define all paths
    const outputCuda = path.join(jobDir, 'output_cuda');
    const outputOpenMP = path.join(jobDir, 'output_openmp');
    const outputMPI = path.join(jobDir, 'output_mpi');

    const exeCuda = path.join(EXE_DIR, 'GPU');
    const exeOpenMP = path.join(EXE_DIR, 'OMP');
    const exeMPI = path.join(EXE_DIR, 'GPU');

    // 3. Create output directories
    await Promise.all([
      fs.mkdir(outputCuda, { recursive: true }),
      fs.mkdir(outputOpenMP, { recursive: true }),
      fs.mkdir(outputMPI, { recursive: true }),
    ]);

    // 4. Define arguments (as per user's v0.2 spec update)
    const argsCuda = [INPUT_DIR, outputCuda, filterType];
    const argsOpenMP = [INPUT_DIR, outputOpenMP, filterType];
    const argsMPI = [INPUT_DIR, outputMPI, filterType];

    // 5. Launch all three processes in parallel and wait for them to finish
    console.log(`[${jobId}] Starting CUDA process...`);
    await runBenchmarkProcess(jobId, exeCuda, argsCuda);
    console.log(`[${jobId}] CUDA finished. Starting OpenMP process...`);
    await runBenchmarkProcess(jobId, exeOpenMP, argsOpenMP);
    console.log(`[${jobId}] OpenMP finished. Starting MPI process...`);
    await runBenchmarkProcess(jobId, exeMPI, argsMPI);
    console.log(`[${jobId}] MPI finished. All processes done.`);

    // 6. All processes succeeded. Read and combine results.
    console.log(`[${jobId}] All processes finished. Combining results.`);
    const [cudaData, openmpData, mpiData] = await Promise.all([
      fs.readFile(path.join(outputCuda, 'time.json'), 'utf8'),
      fs.readFile(path.join(outputOpenMP, 'time.json'), 'utf8'),
      fs.readFile(path.join(outputMPI, 'time.json'), 'utf8'),
    ]);

    const finalData = {
      job_id: jobId,
      cuda_results: JSON.parse(cudaData),
      openmp_results: JSON.parse(openmpData),
      mpi_results: JSON.parse(mpiData),
    };

    // 7. Write final results
    await fs.writeFile(
      path.join(jobDir, '_FINAL_RESULTS.json'),
      JSON.stringify(finalData, null, 2)
    );

    // 8. Update status to "completed"
    await fs.writeFile(statusFile, JSON.stringify({ status: 'completed' }));
    console.log(`[${jobId}] Job completed successfully.`);

  } catch (error) {
    // Handle any failure during the process
    console.error(`[${jobId}] Job failed:`, error.message);
    await fs.writeFile(
      statusFile,
      JSON.stringify({ status: 'failed', error: error.message })
    );
  }
}

// --- API Endpoint 1: Start Benchmark ---
app.post('/api/benchmark/start', async (req, res) => {
  const { filter_type } = req.body;
  if (!filter_type) {
    return res.status(400).send({ error: 'filter_type is required' });
  }

  const jobId = 'job_' + uuidv4();
  const jobDir = path.join(JOBS_DIR, jobId);
  const statusFile = path.join(jobDir, '_STATUS.json');

  try {
    // Create job directory and initial status file
    await fs.mkdir(jobDir, { recursive: true });
    await fs.writeFile(statusFile, JSON.stringify({ status: 'pending' }));

    // Start the background task - DO NOT await it.
    runAllBenchmarks(jobId, filter_type);

    // Immediately return 202 Accepted
    res.status(202).send({ job_id: jobId, status: 'pending' });
  } catch (error) {
    console.error('Failed to create job:', error);
    res.status(500).send({ error: 'Failed to initialize job' });
  }
  // res.status(202).send({ job_id: "job_fcf92def-498e-402e-8746-dee0a3eb434a", status:'processing' });
  
});

// --- API Endpoint 2: Get Job Status (Polling) ---
app.get('/api/benchmark/status/:job_id', async (req, res) => {
  const { job_id } = req.params;
  const statusFile = path.join(JOBS_DIR, job_id, '_STATUS.json');

  try {
    const data = await fs.readFile(statusFile, 'utf8');
    res.status(200).send(JSON.parse(data));
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).send({ error: 'Job not found' });
    } else {
      console.error('Error reading status file:', error);
      res.status(500).send({ error: 'Internal server error' });
    }
  }
});

// --- API Endpoint 3: Get Job Result ---
app.get('/api/benchmark/result/:job_id', async (req, res) => {
  const { job_id } = req.params;
  const statusFile = path.join(JOBS_DIR, job_id, '_STATUS.json');
  const resultsFile = path.join(JOBS_DIR, job_id, '_FINAL_RESULTS.json');

  try {
    // 1. Check status first
    let statusData;
    try {
      statusData = await fs.readFile(statusFile, 'utf8');
    } catch (error) {
      if (error.code === 'ENOENT') {
        return res.status(404).send({ error: 'Job not found' });
      }
      throw error;
    }

    const status = JSON.parse(statusData);

    if (status.status !== 'completed') {
      return res.status(400).send({
        error: `Job status is '${status.status}', not 'completed'.`,
      });
    }

    // 2. Read and return final results
    const resultsData = await fs.readFile(resultsFile, 'utf8');
    res.status(200).send(JSON.parse(resultsData));

  } catch (error) {
    if (error.code === 'ENOENT') {
      // This case means status was "completed" but results file is missing
      console.error(`[${job_id}] CRITICAL: Status is completed but _FINAL_RESULTS.json is missing.`);
      res.status(404).send({ error: 'Results file not found, though job was marked completed.' });
    } else {
      console.error('Error reading results file:', error);
      res.status(500).send({ error: 'Internal server error' });
    }
  }
});

// --- Start Server ---
app.listen(PORT, '0.0.0.0', async () => {
  // Ensure the main 'jobs' directory exists on startup
  try {
    await fs.mkdir(JOBS_DIR, { recursive: true });
    console.log(`Jobs directory ensured at: ${JOBS_DIR}`);
    console.log(`Server running at http://0.0.0.0:${PORT} (listening on all interfaces)`);
  } catch (error) {
    console.error('Failed to create jobs directory:', error);
    process.exit(1);
  }
});
