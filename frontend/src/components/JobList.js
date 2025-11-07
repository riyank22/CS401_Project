import React, { useState, useEffect, useRef } from 'react';
import { API_BASE_URL } from '../config';
import Icon from './Icon';
import Spinner from './Spinner';

// Helper function to format date
const formatDate = (iso) => {
    try {
        const d = new Date(iso);
        return d.toLocaleString();
    } catch {
        return iso;
    }
};

/**
 * JobDisplayBox Component
 * Renders the rich info box for a job.
 * This is the static, presentational part.
 */
function JobDisplayBox({ job }) {
    return (
        <div className="p-4">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center min-w-0"> {/* min-w-0 for truncation */}
                    <Icon name="File" className="w-5 h-5 text-blue-400 mr-2 flex-shrink-0" />
                    <h2 className="text-md font-semibold truncate" title={job.job_id}>
                        {job.job_id}
                    </h2>
                </div>
                <span
                    className={`text-xs font-medium px-2 py-1 rounded flex-shrink-0 ${
                        job.status === 'completed' ? 'bg-green-700 text-green-100' :
                            job.status === 'running' ? 'bg-yellow-700 text-yellow-100' :
                                'bg-gray-700 text-gray-100'
                    }`}
                >
                    {job.status}
                </span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm text-gray-300">
                <div>
                    <div className="text-xs text-gray-400">Operation</div>
                    <div className="mt-1 font-medium text-white capitalize">{job.operation}</div>
                </div>
                <div>
                    <div className="text-xs text-gray-400">Batch Size</div>
                    <div className="mt-1 font-medium text-white">{job.batch_size}</div>
                </div>
                <div className="col-span-2">
                    <div className="text-xs text-gray-400">Last Updated</div>
                    <div className="mt-1 font-medium text-white">{formatDate(job.last_updated)}</div>
                </div>
            </div>
        </div>
    );
}

/**
 * JobOption Component
 * This is the clickable item in the dropdown list.
 * It wraps the JobDisplayBox with hover effects and an onClick handler.
 */
function JobOption({ job, onClick }) {
    return (
        <div
            onClick={onClick}
            className="text-white hover:bg-gray-700 cursor-pointer border-b border-gray-700 last:border-b-0"
        >
            <JobDisplayBox job={job} />
        </div>
    );
}

/**
 * JobList Component
 * Uses a custom dropdown to display rich job data.
 */
function JobList() {
    const [jobs, setJobs] = useState([]);
    const [selectedJobId, setSelectedJobId] = useState(""); // Store just the job ID
    const [isOpen, setIsOpen] = useState(false); // State to manage dropdown visibility
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Ref for detecting clicks outside the dropdown
    const dropdownRef = useRef(null);

    // 1. Fetch the list of jobs
    useEffect(() => {
        const fetchJobs = async () => {
            setLoading(true);
            setError(null);
            try {
                const res = await fetch(`${API_BASE_URL}/api/jobs/list`);
                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.error || 'Failed to fetch jobs');
                }
                const data = await res.json();
                setJobs(data.jobs || []);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchJobs();
    }, []);

    // 2. Effect to handle clicks outside the dropdown to close it
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [dropdownRef]);

    // 3. Handle navigation
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!selectedJobId) {
            setError("Please select a job to load.");
            return;
        }

        setIsSubmitting(true);
        setError(null);
        window.location.href = `/${selectedJobId}`;
    };

    // Find the full job object for the selected ID
    const selectedJobObj = jobs.find((j) => j.job_id === selectedJobId);

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
                    <label htmlFor="job-select-button" className="block text-sm font-medium text-gray-300 mb-2">
                        Select Job
                    </label>

                    {/* THIS IS THE NEW CUSTOM DROPDOWN */}
                    <div className="relative" ref={dropdownRef}>
                        {/* 1. The Trigger Button (now renders the rich box) */}
                        <button
                            type="button"
                            id="job-select-button"
                            onClick={() => setIsOpen(!isOpen)}
                            // We removed all padding from the button itself
                            className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg text-left flex justify-between items-stretch"
                            disabled={loading || isSubmitting}
                        >
                            {/* This container holds the content and grows to fill space */}
                            <div className="flex-grow min-w-0">
                                {selectedJobObj ? (
                                    // If selected, render the rich box
                                    <JobDisplayBox job={selectedJobObj} />
                                ) : (
                                    // If not selected, render placeholder
                                    <div className="p-4 flex items-center h-full"> {/* Use p-4 to match height */}
                                        <span className="truncate text-gray-400">
                                            {loading ? "Loading jobs..." : "Select a job..."}
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* This container just holds the chevron, centered vertically */}
                            <div className="flex items-center px-4 flex-shrink-0 text-gray-400">
                                <Icon name={isOpen ? "ChevronUp" : "ChevronDown"} className="w-5 h-5" />
                            </div>
                        </button>

                        {/* 2. The Dropdown Panel (with rich JobOption components) */}
                        {isOpen && (
                            <div className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg max-h-96 overflow-y-auto">
                                {loading ? (
                                    <div className="p-4 text-gray-400">Loading...</div>
                                ) : (
                                    jobs.map((job) => (
                                        <JobOption
                                            key={job.job_id}
                                            job={job}
                                            onClick={() => {
                                                setSelectedJobId(job.job_id); // Set the selected ID
                                                setIsOpen(false); // Close the dropdown
                                            }}
                                        />
                                    ))
                                )}
                            </div>
                        )}
                    </div>
                </div>

                <button
                    type="submit"
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center disabled:opacity-50 transition-all"
                    disabled={loading || isSubmitting || !selectedJobId}
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