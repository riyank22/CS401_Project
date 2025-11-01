import React from 'react';
import Icon from './Icon';

/**
 * StatCard Component
 * A reusable card for displaying key-value statistics.
 */
function StatCard({ title, value, unit, iconName, loading = false }) {
    return (
        <div className="bg-gray-800 rounded-lg p-5 shadow-lg">
            <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-400">{title}</p>
                <Icon name={iconName} className="w-5 h-5 text-gray-500" />
            </div>
            <div className="mt-2">
                {loading ? (
                    <div className="h-8 w-3/4 bg-gray-700 rounded animate-pulse"></div>
                ) : (
                    <p className="text-3xl font-semibold text-white">
                        {value}
                        <span className="text-lg font-medium text-gray-400 ml-1">{unit}</span>
                    </p>
                )}
            </div>
        </div>
    );
}

export default StatCard;