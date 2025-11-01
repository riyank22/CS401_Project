import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// --- Chart.js Imports ---
// We need to register the components we're going to use.
import {
    Chart,
    BarElement,
    CategoryScale,
    LinearScale,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

Chart.register(
    BarElement,
    CategoryScale,
    LinearScale,
    Title,
    Tooltip,
    Legend
);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);