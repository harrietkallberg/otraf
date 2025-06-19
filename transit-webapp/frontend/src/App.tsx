// src/App.tsx
import React from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import Dashboard from './pages/Dashboard';
import Search from './pages/Search';
import RouteDetail from './pages/RouteDetail';
import StopDetail from './pages/StopDetail';
import Violations from './pages/Violations';
import Analytics from './pages/Analytics';

const App: React.FC = () => {
  const location = useLocation();

  const routeTitles: Record<string, string> = {
    '/': 'Dashboard',
    '/dashboard': 'Dashboard',
    '/search': 'Search',
    '/violations': 'Violations',
    '/analytics': 'Analytics',
  };

  const getPageTitle = (): string => {
    const basePath = location.pathname.split('/')[1];
    if (location.pathname.startsWith('/routes/')) return 'Route Details';
    if (location.pathname.startsWith('/stops/')) return 'Stop Details';
    return routeTitles[`/${basePath}`] || 'Unknown Page';
  };

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 overflow-auto">
        <Topbar title={getPageTitle()} />
        <div className="p-4">
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
    </div>
  );
};

export default App;
