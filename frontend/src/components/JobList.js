import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';
import Icon from './Icon';
import Spinner from './Spinner';

/**
 * JobList Component
 * Fetches a list of previous jobs and navigates to the selected one.
 */
function JobList() {
    const [jobs, setJobs] = useState([]);
    const [selectedJob, setSelectedJob] = useState(""); // Store just the job ID string
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // 1. Fetch the list of jobs when the component mounts
    useEffect(() => {
        const fetchJobs = async () => {
            setLoading(true);
            setError(null);
            try {
                const res = await fetch(`${API_BASE_URL}/api/jobs/list`); // Assumes GET is default

                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.error || 'Failed to fetch jobs');
                }

                const data = await res.json();
                console.log(data.jobs);
                setJobs(data.jobs || []); // Store the array of jobs in state
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchJobs();
    }, []); // <-- Empty array means this runs only ONCE on mount

    // 2. Handle navigation
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!selectedJob) {
            setError("Please select a job to load.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        // No fetch needed! Just navigate.
        // The JobDashboard component will handle fetching for this ID.
        window.location.href = `/${selectedJob}`;
    };

    return (
        <div className="max-w-xl w-full bg-gray-850 p-8 rounded-2xl shadow-2xl border border-gray-700">
            <div className="flex items-center justify-center mb-6">
                <Icon name="History" className="w-10 h-10 text-blue-400 mr-3" />
                <h1 className="text-3xl font-bold text-white">Load Previous Job</h1>
            </div>
            <p className="text-center text-gray-400 mb-8">
                Select a previous job from the list to view its results.
            </p>

            {error && (
                <div className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg mb-6" role="alert">
                    <strong className="font-bold">Error: </strong>
                    <span className="block sm:inline">{error}</span>
                </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <label htmlFor="job-select" className="block text-sm font-medium text-gray-300 mb-2">
                        Select Job ID
                    </label>
                    <select
                        id="job-select"
                        value={selectedJob}
                        onChange={(e) => setSelectedJob(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-4 py-3 focus:ring-blue-500 focus:border-blue-500 transition-all"
                        disabled={loading || isSubmitting}
                    >
                        {/* Default disabled option */}
                        <option value="" disabled>
                            {loading ? "Loading jobs..." : "Select a job..."}
                        </option>

                        {/* 3. Render options by mapping over the 'jobs' state */}
                        {jobs.map((job) => (
                            <option key={job} value={job}>
                                {job}
                            </option>
                        ))}
                    </select>
                </div>

                <button
                    type="submit"
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center disabled:opacity-50 transition-all"
                    disabled={loading || isSubmitting || !selectedJob}
                >
                    {isSubmitting ? (
                        <>
                            <Spinner size="h-5 w-5 mr-2" />
                            Loading...
                        </>
                    ) : (
                        <>
                            <Icon name="Search" className="w-5 h-5 mr-2" />
                            Load Job
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}

export default JobList;