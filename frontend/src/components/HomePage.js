import React from 'react';
import Uploader from './Uploader';
import JobList from './JobList';

function HomePage() {
    return (
        <div className="min-h-full flex flex-col items-center justify-center p-4 space-y-8 lg:space-y-0 lg:flex-row lg:space-x-8 lg:items-start">
            {/* Box 1: New Job */}
            <Uploader />

            {/* Divider */}
            <div className="flex items-center justify-center text-gray-400 font-bold text-2xl h-full lg:h-auto py-4 lg:py-0">
                OR
            </div>

            {/* Box 2: Load Job */}
            <JobList />
        </div>
    );
}

export default HomePage;