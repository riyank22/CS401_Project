import React from 'react';

/**
 * StatusIndicator Component
 * Shows a colored dot and text for job status.
 */
function StatusIndicator({ status }) {
    const baseClass = 'status-dot';
    let statusClass = 'status-pending';
    let text = 'Pending';

    switch (status) {
        case 'processing':
            statusClass = 'status-processing';
            text = 'Processing';
            break;
        case 'completed':
            statusClass = 'status-completed';
            text = 'Completed';
            break;
        case 'failed':
            statusClass = 'status-failed';
            text = 'Failed';
            break;
        default:
            // Keep default pending state
            break;
    }

    return (
        <div className="flex items-center">
            <span className={`${baseClass} ${statusClass}`}></span>
            <span className="capitalize text-sm font-medium">{text}</span>
        </div>
    );
}

export default StatusIndicator;