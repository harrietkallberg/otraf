import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useEffect, useState } from 'react';
import Dashboard from './pages/Dashboard';
import StopCentric from './pages/StopCentric';
import RouteCentric from './pages/RouteCentric';
import NotFound from './pages/NotFound';

function App() {
  const [routes, setRoutes] = useState<string[]>([]);
  const [globalFiles, setGlobalFiles] = useState<string[]>([]);

  useEffect(() => {
    // Fetch available route folders
    fetch('/api/routes')
      .then((res) => res.json())
      .then((data) => setRoutes(data))
      .catch((err) => console.error('Failed to fetch routes', err));

    // Fetch available global files
    fetch('/api/global')
      .then((res) => res.json())
      .then((data) => setGlobalFiles(data))
      .catch((err) => console.error('Failed to fetch global files', err));
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard routes={routes} globalFiles={globalFiles} />} />
        <Route path="/stop/:stopId" element={<StopCentric />} />
        <Route path="/route/:routeName" element={<RouteCentric />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

export default App;