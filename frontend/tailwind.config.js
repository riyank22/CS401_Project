/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            colors: {
                gray: {
                    // You need to manually define the full gray scale
                    // or import from 'tailwindcss/colors'
                    ...require('tailwindcss/colors').gray,
                    850: '#182030',
                    950: '#0b0f1a'
                }
            }
        },
    },
    plugins: [],
}