import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Search from './pages/Search';
import RouteDetail from './pages/RouteDetail';
import StopDetail from './pages/StopDetail';
import Violations from './pages/Violations';
import Analytics from './pages/Analytics';

const App: React.FC = () => {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 overflow-auto p-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/search" element={<Search />} />
          <Route path="/routes/:routeId" element={<RouteDetail />} />
          <Route path="/stops/:stopId" element={<StopDetail />} />
          <Route path="/violations" element={<Violations />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </div>
    </div>
  );
};

export default App;
