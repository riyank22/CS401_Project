import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';
import Spinner from './Spinner';
import ImageModal from './ImageModal';
import StatusIndicator from './StatusIndicator';
import StatCard from './StatCard';
import BenchmarkChart from './BenchmarkChart';
import ImageGrid from './ImageGrid';
import Icon from "./Icon";

/**
 * JobDashboard Component
 * The main view for a running or completed job.
 * Handles all polling and data fetching for the dashboard.
 */
function JobDashboard({ jobId }) {
    const [status, setStatus] = useState(null);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [modalImage, setModalImage] = useState(null);
    const [operation, setOperation] = useState(null);

    // 1. Polling useEffect for /status
    useEffect(() => {
        if (!jobId) return;

        let intervalId;

        const poll = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/api/jobs/status/${jobId}`);
                if (!res.ok) {
                    throw new Error('Job not found');
                }
                const data = await res.json();
                setStatus(data);
                setOperation(data.operation);

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(intervalId);
                }
            } catch (err) {
                setError(err.message);
                clearInterval(intervalId);
            }
        };

        poll(); // Initial poll
        intervalId = setInterval(poll, 3000); // Poll every 3 seconds

        return () => clearInterval(intervalId);
    }, [jobId]);

    // 2. Data Fetching useEffect for /result (triggers on status change)
    useEffect(() => {
        if (!status || status.status === 'pending') {
            return; // Don't fetch full results yet
        }

        // Don't refetch if we already have the data for this status
        // Use optional chaining and remove 'result' from dependency array
        // to prevent potential re-fetch loops.
        if (result?.status_details?.last_updated === status.last_updated) {
            return;
        }

        const fetchResult = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/api/jobs/result/${jobId}`);
                if (!res.ok) {
                    throw new Error('Failed to fetch job results');
                }
                const data = await res.json();
                setResult(data);
            } catch (err) {
                setError(err.message);
            }
        };

        fetchResult();
    }, [status, jobId]); // <-- REMOVED 'result' from dependencies

    if (error) {
        return (
            <div className="min-h-full flex items-center justify-center p-4">
                <div className="bg-red-900 border border-red-700 text-red-100 px-6 py-4 rounded-lg" role="alert">
                    <strong className="font-bold text-lg">Job Error</strong>
                    <p>{error}</p>
                    <a href="/" className="mt-2 inline-block text-white underline">Go Home</a>
                </div>
            </div>
        );
    }

    if (!status) {
        return (
            <div className="min-h-full flex items-center justify-center">
                <Spinner size="h-10 w-10" />
                <p className="text-xl ml-4">Loading Job Dashboard...</p>
            </div>
        );
    }

    const isLoading = !result;
    const jobStatus = status.status;

    // --- START: MODIFIED LOGIC FOR AVERAGES ---

    let totalLoad = 0;
    let loadCount = 0;
    let totalExport = 0;
    let exportCount = 0;

    // Create an array of potential data sources
    const dataSources = [result?.cuda_data, result?.openmp_data, result?.mpi_data];

    // Iterate over the sources and add to totals if they exist
    dataSources.forEach(data => {
        // Check for loading time
        if (data && typeof data.total_loading_time === 'number') {
            totalLoad += data.total_loading_time;
            loadCount++;
        }
        // Check for exporting time
        if (data && typeof data.total_exporting_time === 'number') {
            totalExport += data.total_exporting_time;
            exportCount++;
        }
    });

    // Calculate averages, handling division by zero
    const avgLoadTime = loadCount > 0 ? totalLoad / loadCount : null;
    const avgExportTime = exportCount > 0 ? totalExport / exportCount : null;

    // --- END: MODIFIED LOGIC FOR AVERAGES ---


    return (
        <div className="min-h-full p-4 sm:p-8 bg-gray-900">
            {modalImage && result && (
                <ImageModal
                    job_id={jobId}
                    filename={modalImage}
                    result={result}
                    onClose={() => setModalImage(null)}
                />
            )}

            <header className="mb-8">
                <div className="flex flex-col sm:flex-row justify-between sm:items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Job Dashboard</h1>
                        <p className="text-sm text-gray-400 break-all">ID: {jobId}</p>
                    </div>
                    <div className="mt-4 sm:mt-0">
                        <a href="/" className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
                            <Icon name="Plus" className="w-5 h-5 mr-2" />
                            New Benchmark
                        </a>
                    </div>
                </div>

                {jobStatus === 'failed' && (
                    <div className="mt-4 bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg" role="alert">
                        <strong className="font-bold">Job Failed: </strong>
                        <span className="block sm:inline">{status.error || 'An unknown error occurred.'}</span>
                    </div>
                )}

                {jobStatus === 'completed' && (
                    <div className="mt-4 bg-green-900 border border-green-700 text-green-100 px-4 py-3 rounded-lg" role="alert">
                        <span className="block sm:inline">All benchmarks finished.</span>
                    </div>
                )}
            </header>

            <main className="space-y-6">
                {/* --- Status & Summary Cards --- */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {/* Status Cards */}
                    <div className="bg-gray-800 rounded-lg p-5 shadow-lg space-y-4">
                        <h3 className="text-lg font-semibold text-white">Compute Engine Status</h3>
                        <StatusIndicator computeName={"CUDA"} status={status.cuda} />
                        <StatusIndicator computeName={"OpenMP"} status={status.openmp} />
                        <StatusIndicator computeName={"MPI"} status={status.mpi} />
                    </div>

                    {/* Summary Cards */}
                    <StatCard
                        title="Batch Size"
                        value={isLoading ? '...' : result.batch_size }
                        unit=" Images"
                        loading={isLoading}
                    />

                    {/* --- START: MODIFIED STAT CARDS --- */}
                    <StatCard
                        title="Average Load Time"
                        value={avgLoadTime ? avgLoadTime.toFixed(2) : '...'}
                        unit="ms"
                        iconName="DownloadCloud"
                        loading={isLoading}
                    />
                    <StatCard
                        title="Average Export Time"
                        value={avgExportTime ? avgExportTime.toFixed(2) : '...'}
                        unit="ms"
                        iconName="UploadCloud"
                        loading={isLoading}
                    />
                    {/* --- END: MODIFIED STAT CARDS --- */}
                </div>

                {/* --- Main Chart --- */}
                <BenchmarkChart result={result} />

                {/* --- Image Grid --- */}
                <ImageGrid result={result} onImageClick={setModalImage} />

            </main>
        </div>
    );
}

export default JobDashboard;