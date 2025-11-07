import React from 'react';
import Uploader from './Uploader';
import JobList from './JobList';

function HomePage() {
        return (
            <div className="min-h-full flex flex-col items-center justify-around p-4 gap-8 lg:flex-row lg:justify-around lg:items-start w-full">
                <div className="flex-1 max-w-md w-full">
                    <Uploader />
                </div>

                <div className="flex-1 max-w-md w-full">
                    <JobList/>
                </div>
            </div>
        );
    }

export default HomePage;