import React from 'react';
import { icons } from 'lucide-react';

/**
 * Icon Component
 * Dynamically renders a Lucide icon.
 */
const Icon = ({ name, ...props }) => {
    const IconComponent = icons[name];

    if (!IconComponent) {
        console.error(`Lucide icon "${name}" not found`);
        return null;
    }

    return <IconComponent {...props} />;
};

export default Icon;