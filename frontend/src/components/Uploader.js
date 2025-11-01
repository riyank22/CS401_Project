import React, { useState, useMemo } from 'react';
import { API_BASE_URL } from '../config';
import Icon from './Icon';
import Spinner from './Spinner';

/**
 * Uploader (Main Page Component)
 * Handles file selection, filter choice, and job submission.
 */
function Uploader() {
    const [files, setFiles] = useState([]);
    const [filter, setFilter] = useState('grayscale');
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setFiles([...e.target.files]);
        setError(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (files.length === 0) {
            setError('Please select at least one image file.');
            return;
        }

        setIsUploading(true);
        setError(null);

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        formData.append('filter_type', filter);

        try {
            const res = await fetch(`${API_BASE_URL}/api/jobs/submit`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || 'Upload failed');
            }

            const data = await res.json();
            // Navigate to the job dashboard page
            window.location.href = `/${data.job_id}`;

        } catch (err) {
            setError(err.message);
            setIsUploading(false);
        }
    };

    const fileNames = useMemo(() => files.map(f => f.name).join(', '), [files]);

    return (
        <div className="min-h-full flex items-center justify-center p-4">
            <div className="max-w-xl w-full bg-gray-850 p-8 rounded-2xl shadow-2xl border border-gray-700">
                <div className="flex items-center justify-center mb-6">
                    <Icon name="DatabaseZap" className="w-10 h-10 text-blue-400 mr-3" />
                    <h1 className="text-3xl font-bold text-white">Benchmark Uploader</h1>
                </div>
                <p className="text-center text-gray-400 mb-8">
                    Upload your images to compare processing performance across CUDA, OpenMP, and MPI.
                </p>

                {error && (
                    <div className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg mb-6" role="alert">
                        <strong className="font-bold">Error: </strong>
                        <span className="block sm:inline">{error}</span>
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label htmlFor="filter" className="block text-sm font-medium text-gray-300 mb-2">
                            Select Filter
                        </label>
                        <select
                            id="filter"
                            value={filter}
                            onChange={(e) => setFilter(e.target.value)}
                            className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-4 py-3 focus:ring-blue-500 focus:border-blue-500 transition-all"
                            disabled={isUploading}
                        >
                            <option value="grayscale">Grayscale</option>
                            <option value="sobel">Sobel</option>
                            <option value="gaussian">Gaussian</option>
                        </select>
                    </div>

                    <div>
                        <label htmlFor="file-upload" className="block text-sm font-medium text-gray-300 mb-2">
                            Upload Images
                        </label>
                        <label
                            htmlFor="file-upload"
                            className={`relative cursor-pointer bg-gray-900 rounded-lg border-2 border-dashed border-gray-700 p-6 flex flex-col items-center justify-center text-gray-400 hover:border-gray-500 transition-all ${isUploading ? 'opacity-50' : ''}`}
                        >
                            <Icon name="UploadCloud" className="w-12 h-12 mb-2" />
                            <span className="font-semibold text-blue-400">Click to upload</span>
                            <span className="text-sm">or drag and drop</span>
                            <input
                                id="file-upload"
                                name="files"
                                type="file"
                                multiple
                                accept="image/png, image/jpeg, image/jpg"
                                className="sr-only"
                                onChange={handleFileChange}
                                disabled={isUploading}
                            />
                        </label>
                        {files.length > 0 && (
                            <div className="mt-3 text-sm text-gray-400">
                                <strong>Selected:</strong> {fileNames}
                            </div>
                        )}
                    </div>

                    <button
                        type="submit"
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center disabled:opacity-50 transition-all"
                        disabled={isUploading}
                    >
                        {isUploading ? (
                            <React.Fragment>
                                <Spinner size="h-5 w-5 mr-2" />
                                Uploading...
                            </React.Fragment>
                        ) : (
                            <React.Fragment>
                                <Icon name="Play" className="w-5 h-5 mr-2" />
                                Start Benchmark
                            </React.Fragment>
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
}

export default Uploader;