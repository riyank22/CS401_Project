import React from 'react';
import { Bar } from 'react-chartjs-2';
import Spinner from './Spinner';

/**
 * BenchmarkChart Component
 * Renders the bar chart for processing times.
 */
function BenchmarkChart({ result }) {

    const chartData = {
        labels: ['CUDA', 'OpenMP', 'MPI'],
        datasets: [
            {
                label: 'Total Processing Time (ms)',
                data: [
                    result?.cuda_data?.total_processing_time || 0,
                    result?.openmp_data?.total_processing_time || 0,
                    result?.mpi_data?.total_processing_time || 0,
                ],
                backgroundColor: [
                    'rgba(110, 231, 183, 0.5)', // CUDA (green)
                    'rgba(96, 165, 250, 0.5)',  // OpenMP (blue)
                    'rgba(240, 171, 252, 0.5)', // MPI (purple)
                ],
                borderColor: [
                    'rgb(110, 231, 183)',
                    'rgb(96, 165, 250)',
                    'rgb(240, 171, 252)',
                ],
                borderWidth: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            title: {
                display: true,
                text: 'Engine Performance Comparison',
                color: '#e5e7eb',
                font: { size: 16 }
            },
            tooltip: {
                callbacks: {
                    label: (context) => `${context.raw.toFixed(2)} ms`
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Time (ms)',
                    color: '#9ca3af'
                },
                ticks: { color: '#9ca3af' },
                grid: { color: '#374151' }
            },
            x: {
                ticks: { color: '#9ca3af' },
                grid: { color: '#374151' }
            }
        }
    };

    const isLoading = !result || (result.status_details.status !== 'completed' && !result.cuda_data);

    return (
        <div className="bg-gray-850 p-6 rounded-lg shadow-lg border border-gray-700 h-96">
            {isLoading ? (
                <div className="h-full flex items-center justify-center flex-col">
                    <Spinner size="h-10 w-10" />
                    <p className="text-gray-400 mt-4">Waiting for first result...</p>
                </div>
            ) : (
                <Bar options={options} data={chartData} />
            )}
        </div>
    );
}

export default BenchmarkChart;