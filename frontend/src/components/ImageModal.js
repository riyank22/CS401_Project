import React, { useState, useEffect } from 'react';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { API_BASE_URL } from '../config';
import Icon from './Icon';
import Spinner from './Spinner';

/**
 * Formats a time in milliseconds to either seconds (s) or milliseconds (ms).
 * @param {number | null | undefined} ms - The time in milliseconds.
 * @returns {string} - The formatted string (e.g., "1.23 s" or "456.78 ms").
 */
const formatTime = (ms) => {
    // Handle null, undefined, or 0
    if (!ms) {
        return '...';
    }

    if (ms >= 1000) {
        // Convert to seconds
        return `${(ms / 1000).toFixed(2)} s`;
    }

    // Keep in milliseconds
    return `${ms.toFixed(2)} ms`;
};

// --- ADDITION 1: Helper function to calculate average stats for a single image ---
// This is the same powerful helper we used in ImageGrid.
const getImageStats = (result, filename) => {
    let totalLoad = 0, loadCount = 0;
    let totalExport = 0, exportCount = 0;

    const dataSources = [result?.cuda_data, result?.openmp_data, result?.mpi_data];

    dataSources.forEach(data => {
        if (data && data.individual_image_times) {
            const imgData = data.individual_image_times.find(t => t.image_name === filename);
            if (imgData) {
                if (typeof imgData.load_ms === 'number') { totalLoad += imgData.load_ms; loadCount++; }
                if (typeof imgData.export_ms === 'number') { totalExport += imgData.export_ms; exportCount++; }
            }
        }
    });

    return {
        avgLoad: loadCount > 0 ? totalLoad / loadCount : null,
        avgExport: exportCount > 0 ? totalExport / exportCount : null,
    };
};

// --- ADDITION 2: A small component to display a single stat nicely ---
const StatDisplay = ({ label, value, unit }) => (
    <div className="text-center sm:text-right">
        <p className="text-xs text-gray-400 font-medium">{label}</p>
        <p className="text-lg font-semibold text-white whitespace-nowrap">
            {value !== null ? `${value.toFixed(2)} ${unit}` : '...'}
        </p>
    </div>
);


/**
 * ImageModal Component
 * The full-screen "drill-down" view with SYNCHRONIZED 4-way zoom and pan.
 */
function ImageModal({ filename, result, onClose }) {
    const [transform, setTransform] = useState({
        scale: 1,
        positionX: 0,
        positionY: 0,
    });

    const handleTransform = (ref) => {
        setTransform(ref.state);
    };

    const inputImage = result.input_images.find(img => img.filename === filename);

    // --- ADDITION 3: Calculate the average stats for the current image ---
    const imageStats = getImageStats(result, filename);

    const getImageData = (engineName, data) => {
        if (!data || !inputImage) return { url: null, time: null, status: result.status_details[engineName] };
        const imageTimeData = data.individual_image_times?.find(t => t.image_name === filename);
        return {
            url: `${API_BASE_URL}${inputImage[`${engineName}_output_url`]}`,
            time: imageTimeData?.process_ms,
            status: 'completed'
        }
    };

    const engines = [
        { name: 'Original', data: { url: `${API_BASE_URL}${inputImage?.url}`, time: null, status: 'completed' } },
        { name: 'CUDA', data: getImageData('cuda', result.cuda_data) },
        { name: 'OpenMP', data: getImageData('openmp', result.openmp_data) },
        { name: 'MPI', data: getImageData('mpi', result.mpi_data) },
    ];

    useEffect(() => {
        const handleEsc = (event) => {
            if (event.keyCode === 27) onClose();
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [onClose]);

    // renderImageTile function remains unchanged...
    const renderImageTile = (engine) => {
        const { url, time, status } = engine.data;
        let content;

        // This logic defines WHAT to show (image, spinner, or text)
        if (status === 'completed' && url) {
            content = (
                <img
                    src={url}
                    alt={`${engine.name} processed - ${filename}`}
                    className="w-full h-full object-contain" // Image will fill the "contain" area
                    onError={(e) => {
                        e.target.src = 'https://placehold.co/400x400/1f2937/9ca3af?text=Image+Error';
                        console.error(`Failed to load image: ${url}`);
                    }}
                />
            );
        } else if (status === 'processing') {
            content = <div className="flex flex-col items-center justify-center h-full"><Spinner /><p className="mt-2 text-sm text-gray-400">Processing...</p></div>;
        } else if (status === 'failed') {
            content = <div className="flex items-center justify-center h-full text-red-400 font-bold">Failed</div>;
        }
        else {
            content = <div className="flex items-center justify-center h-full text-gray-500">Pending</div>;
        }

        // This is the tile structure
        return (
            <div key={engine.name} className="flex flex-col bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
                {/* Header */}
                <div className="p-3 border-b border-gray-700 bg-gray-800 flex justify-between items-center">
                    <h4 className="font-semibold text-white">{engine.name}</h4>
                    <p className="text-sm text-gray-400">Time:
                        <span className="font-medium text-white ml-1">
              {time ? `${time.toFixed(2)} ms` : (engine.name === 'Original' ? 'N/A' : '...')}
            </span>
                    </p>
                </div>

                {/* UPDATED: Zoomable Area */}
                <div className="flex-grow flex items-center justify-center p-2 bg-gray-950 overflow-hidden">
                    <TransformWrapper
                        // Controlled component: Read state from parent
                        scale={transform.scale}
                        positionX={transform.positionX}
                        positionY={transform.positionY}
                        // Update parent state on change
                        onTransformed={handleTransform}
                        // Configuration
                        limitToBounds={true}
                        doubleClick={{ mode: "reset" }} // Double-click to reset zoom
                        wheel={{ step: 0.2 }}
                        wrapperClass="w-full h-full"
                    >
                        <TransformComponent
                            // These styles ensure the content (image or spinner) is centered
                            wrapperStyle={{ width: "100%", height: "100%" }}
                            contentStyle={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}
                        >
                            {content} {/* This is our image, spinner, or text */}
                        </TransformComponent>
                    </TransformWrapper>
                </div>
            </div>
        );
    };

    if (!inputImage) {
        return (
            <div
                className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
                onClick={onClose}
            >
                <p className="text-white">Error: Could not find image data.</p>
            </div>
        );
    }

    return (
        <div
            className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50 p-4 sm:p-8 overflow-auto"
            onClick={onClose}
        >
            <div
                className="bg-gray-850 rounded-2xl shadow-2xl p-4 w-full h-full max-h-[95vh] border border-gray-700 flex flex-col"
                onClick={(e) => e.stopPropagation()}
            >
                {/* --- MODIFIED HEADER --- */}
                <div className="flex justify-between items-center mb-4 pb-4 border-b border-gray-700 flex-wrap gap-4">
                    {/* Left side: Title and filename */}
                    <div className="flex-shrink-0">
                        <h2 className="text-2xl font-bold text-white">Resultant Images</h2>
                        {/* Added truncation for very long filenames */}
                        <p className="text-gray-400 truncate max-w-xs sm:max-w-sm md:max-w-md" title={filename}>{filename}</p>
                    </div>

                    {/* Right side: Average Stats and Close Button */}
                    <div className="flex items-center gap-6 sm:gap-8">
                        <StatDisplay label="Avg. Load" value={imageStats.avgLoad} unit="ms" />
                        <StatDisplay label="Avg. Export" value={imageStats.avgExport} unit="ms" />
                        <button
                            onClick={onClose}
                            className="text-gray-500 hover:text-white transition-colors p-1"
                        >
                            <Icon name="X" className="w-8 h-8" />
                        </button>
                    </div>
                </div>
                {/* --- END MODIFIED HEADER --- */}

                {/* 2x2 Image Grid */}
                <div className="flex-grow grid grid-cols-1 md:grid-cols-2 gap-4 overflow-auto">
                    {engines.map(renderImageTile)}
                </div>
            </div>
        </div>
    );
}

export default ImageModal;