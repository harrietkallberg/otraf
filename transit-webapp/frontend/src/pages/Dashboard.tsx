// src/pages/Dashboard.tsx
import React, { useEffect, useState } from 'react';

interface SummaryStats {
  totalRoutes: number;
  totalStops: number;
  totalViolations: number;
  totalDirections: number;
}

// Simulate API loader (replace with real fetch call later)
const loadRouteSummaries = async (): Promise<SummaryStats> => {
  return {
    totalRoutes: 34,
    totalStops: 276,
    totalViolations: 89,
    totalDirections: 61
  };
};

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<SummaryStats | null>(null);

  useEffect(() => {
    loadRouteSummaries().then(setStats);
  }, []);

  if (!stats) return <div>Loading summary...</div>;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">📊 System Summary</h1>
      <div className="dashboard-summary">
        <div><strong>Routes:</strong> {stats.totalRoutes}</div>
        <div><strong>Stops:</strong> {stats.totalStops}</div>
        <div><strong>Directions:</strong> {stats.totalDirections}</div>
        <div><strong>Violations:</strong> {stats.totalViolations}</div>
      </div>
    </div>
  );
};

export default Dashboard;
