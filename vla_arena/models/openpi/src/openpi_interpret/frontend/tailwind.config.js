/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        cmu: {
          red: '#C41230',
          black: '#000000',
          'iron-gray': '#6D6E71',
          'steel-gray': '#E0E0E0',
          'brick-beige': '#E4DAC4',
          'tan': '#BCB49E',
          'hornbostel-teal': '#1F4C4C',
          'palladian-green': '#719F94',
          'weaver-blue': '#182C4B',
          'skibo-red': '#941120',
          'gold-thread': '#FDB515',
          'green-thread': '#009647',
          'blue-thread': '#043673',
          'sky-blue': '#007BC0',
          'teal-thread': '#008F91',
        },
      },
    },
  },
  plugins: [],
}
