import React from 'react';

export default function StatusBadge({ status }) {
  const colors = {
    pending: '#6b7280',
    processing: '#3b82f6',
    completed: '#22c55e',
    failed: '#ef4444',
  };

  const text = status.charAt(0).toUpperCase() + status.slice(1);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: '50%',
          backgroundColor: colors[status] || '#6b7280',
        }}
      ></span>
      <span>{text}</span>
    </div>
  );
}
