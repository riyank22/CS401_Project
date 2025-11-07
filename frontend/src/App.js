import React, {useEffect} from 'react';
import JobDashboard from './components/JobDashboard';
import HomePage from "./components/HomePage";

/**
 * App Component (Main Entry Point)
 * Acts as a simple router based on the URL path.
 */
function App() {
    const path = window.location.pathname;
    const jobId = path.substring(1); // Get string after "/"

    if (jobId) {
        // We are on a job page
        return <JobDashboard jobId={jobId} />;
    }

    // We are on the home page
    return <HomePage />;
}

export default App;