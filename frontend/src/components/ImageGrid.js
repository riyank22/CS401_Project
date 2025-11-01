import React from 'react';
import { API_BASE_URL } from '../config';

/**
 * ImageGrid Component
 * Displays a grid of the original input images and their per-engine times.
 */
function ImageGrid({ result, onImageClick }) {
    if (!result || !result.input_images || result.input_images.length === 0) {
        return (
            <div className="bg-gray-850 p-6 rounded-lg shadow-lg border border-gray-700 text-center">
                <p className="text-gray-400">Waiting for images...</p>
            </div>
        );
    }

    const getImageTime = (data, filename) => {
        if (!data || !data.individual_image_times) return null;
        const imgData = data.individual_image_times.find(t => t.image_name === filename);
        return imgData?.process_ms;
    }

    const TimeDisplay = ({ time }) => (
        <span className="text-sm font-medium text-white">
      {time ? `${time.toFixed(2)} ms` : '...'}
    </span>
    );

    return (
        <div className="bg-gray-850 p-6 rounded-lg shadow-lg border border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-4">Image Analysis</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {result.input_images.map(image => {
                    const cudaTime = getImageTime(result.cuda_data, image.filename);
                    const openmpTime = getImageTime(result.openmp_data, image.filename);
                    const mpiTime = getImageTime(result.mpi_data, image.filename);

                    return (
                        <div
                            key={image.filename}
                            className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700 cursor-pointer hover:border-blue-500 hover:shadow-xl transition-all"
                            onClick={() => onImageClick(image.filename)}
                        >
                            <img
                                src={`${API_BASE_URL}${image.url}`}
                                alt={`Input: ${image.filename}`}
                                className="w-full h-40 object-cover"
                                loading="lazy"
                                onError={(e) => e.target.src = 'https://placehold.co/400x300/1f2937/9ca3af?text=Image+Error'}
                            />
                            <div className="p-3">
                                <p className="text-sm font-medium text-white truncate" title={image.filename}>{image.filename}</p>
                                <div className="text-xs text-gray-400 mt-2 space-y-1">
                                    <p>CUDA: <TimeDisplay time={cudaTime} /></p>
                                    <p>OpenMP: <TimeDisplay time={openmpTime} /></p>
                                    <p>MPI: <TimeDisplay time={mpiTime} /></p>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export default ImageGrid;