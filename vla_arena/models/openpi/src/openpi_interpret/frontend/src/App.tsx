import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { HomePage } from './components/HomePage';
import { EpisodePage } from './components/EpisodePage';

/** Root application component with client-side routing. */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/episode/:episodeId" element={<EpisodePage />} />
      </Routes>
    </BrowserRouter>
  );
}
