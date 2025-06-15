import React, { useState, useEffect } from 'react';

// Types matching your adjusted backend + JSON structure
interface RouteData {
  canonical_pattern: Record<string, { stop_name: string; complete?: boolean }>;
  observed_patterns?: any;
}

interface StopViolation {
  stop_name: string;
  route_name: string;
  stop_type: string;
  severity: string;
  violation_type: string;
  description: string;
}

interface SystemStats {
  directionNavigation: Record<string, RouteData>;
  stopViolations: {
    parent_station_violations: Record<string, StopViolation>;
    metadata: {
      total_violations: number;
    };
  };
}

const App: React.FC = () => {
  const [selectedView, setSelectedView] = useState('overview');
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const [directionNavRes, stopViolationsRes] = await Promise.all([
          fetch('http://localhost:5000/api/direction-navigation'),
          fetch('http://localhost:5000/api/stop-violations')
        ]);

        if (!directionNavRes.ok || !stopViolationsRes.ok) {
          throw new Error('One or more endpoints failed');
        }

        const directionNavigation = await directionNavRes.json();
        const stopViolations = await stopViolationsRes.json();

        setSystemStats({ directionNavigation, stopViolations });
      } catch (err: any) {
        console.error(err);
        setError(err.message || 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div style={{ padding: 20 }}>Loading dashboard...</div>;
  if (error) return <div style={{ color: 'red', padding: 20 }}>Error: {error}</div>;
  if (!systemStats) return <div style={{ padding: 20 }}>No data loaded.</div>;

  const directionNavigation = systemStats.directionNavigation;
  const stopViolations = systemStats.stopViolations.parent_station_violations;

  const getStatusColor = (severity: string) => {
    const color = {
      normal: '#10B981',
      minor: '#F59E0B',
      severe: '#EF4444'
    }[severity.toLowerCase()] || '#6B7280';
    return { color };
  };

  const OverviewView = () => {
    const totalRoutes = Object.keys(directionNavigation).length;
    const totalStops = Object.values(directionNavigation).reduce(
      (sum, route) => sum + Object.keys(route.canonical_pattern || {}).length, 0
    );

    return (
      <div style={{ padding: 20 }}>
        <h2>System Overview</h2>
        <p>Total Routes: {totalRoutes}</p>
        <p>Total Stops: {totalStops}</p>
        <p>Violations: {Object.keys(stopViolations).length}</p>

        <h3>Routes</h3>
        {Object.entries(directionNavigation).map(([routeId, routeData]) => (
          <div key={routeId}
            style={{ marginBottom: 10, padding: 10, border: '1px solid #ddd', borderRadius: 4, cursor: 'pointer' }}
            onClick={() => {
              setSelectedRoute(routeId);
              setSelectedView('route');
            }}
          >
            <strong>Route {routeId}</strong> — {Object.keys(routeData.canonical_pattern).length} stops
          </div>
        ))}
      </div>
    );
  };

  const RouteView = () => {
    if (!selectedRoute) return null;
    const routeData = directionNavigation[selectedRoute];
    const stops = routeData?.canonical_pattern || {};

    return (
      <div style={{ padding: 20 }}>
        <button onClick={() => setSelectedView('overview')}>← Back</button>
        <h2>Route {selectedRoute}</h2>
        {Object.entries(stops).map(([seq, stop]) => {
          const violationEntry = Object.entries(stopViolations).find(([_, v]) => v.stop_name === stop.stop_name)?.[1];
          const severity = violationEntry?.severity || 'Normal';

          return (
            <div key={seq}
              style={{
                padding: '10px',
                margin: '6px 0',
                border: '1px solid #ddd',
                borderLeft: `6px solid ${getStatusColor(severity).color}`,
                borderRadius: '4px'
              }}
            >
              <strong>{stop.stop_name}</strong> <span style={getStatusColor(severity)}>{severity}</span>
              {violationEntry?.description && <p style={{ margin: '5px 0' }}>{violationEntry.description}</p>}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif' }}>
      <header style={{ background: '#f3f4f6', padding: 20, borderBottom: '1px solid #ccc' }}>
        <h1 style={{ margin: 0 }}>Transit Analysis Dashboard</h1>
      </header>
      <main style={{ maxWidth: 800, margin: '0 auto' }}>
        {selectedView === 'overview' && <OverviewView />}
        {selectedView === 'route' && <RouteView />}
      </main>
    </div>
  );
};

export default App;
