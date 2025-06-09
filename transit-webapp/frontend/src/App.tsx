import React, { useState, useEffect } from 'react';

// Type definitions matching your JSON structure
interface RouteData {
  route_name: string;
  stops: string[];
  total_stops: number;
  summary: {
    normal_stops_count: number;
    minor_stops_count: number;
    severe_stops_count: number;
  };
}

interface StopAnalysis {
  stop_name: string;
  route_long_name: string;
  stop_type: string;
  problematic_status: string;
  problematic_type: string;
  description: string;
}

interface SystemStats {
  route_stops: Record<string, RouteData>;
  stop_analysis: Record<string, StopAnalysis>;
}

const App: React.FC = () => {
  const [selectedView, setSelectedView] = useState<string>('overview');
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Data states - pure data from backend
  const [routeStopsData, setRouteStopsData] = useState<Record<string, RouteData>>({});
  const [stopAnalysisData, setStopAnalysisData] = useState<Record<string, StopAnalysis>>({});
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);

  // Fetch data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Test backend connection first
        const healthResponse = await fetch('http://localhost:5000/');
        const healthData = await healthResponse.json();
        console.log('Backend connected:', healthData);

        // Fetch all required data - backend does all calculations
        const [routesResponse, analysisResponse, statsResponse] = await Promise.all([
          fetch('http://localhost:5000/api/routes'),
          fetch('http://localhost:5000/api/analysis'),
          fetch('http://localhost:5000/api/stats')
        ]);

        if (!routesResponse.ok || !analysisResponse.ok || !statsResponse.ok) {
          throw new Error('Failed to fetch data from backend');
        }

        const routesData = await routesResponse.json();
        const analysisData = await analysisResponse.json();
        const statsData = await statsResponse.json();

        console.log('Raw data received:');
        console.log('Routes:', routesData);
        console.log('Analysis:', analysisData);
        console.log('Stats:', statsData);

        // Just store the data - no processing needed
        setRouteStopsData(routesData);
        setStopAnalysisData(analysisData);
        setSystemStats(statsData);

        console.log('Data loaded successfully');
        console.log('Routes:', Object.keys(routesData).length);
        console.log('Stop analyses:', Object.keys(analysisData).length);
        console.log('Stats:', statsData);

      } catch (err: any) {
        console.error('Error fetching data:', err);
        setError(err.message || 'Failed to connect to backend');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getStatusColor = (status: string): React.CSSProperties => {
    switch(status?.toLowerCase()) {
      case 'normal': return { color: '#10B981' };
      case 'minor': return { color: '#F59E0B' };
      case 'severe': return { color: '#EF4444' };
      default: return { color: '#6B7280' };
    }
  };

  const getStatusIcon = (status: string): string => {
    switch(status?.toLowerCase()) {
      case 'normal': return '✓';
      case 'minor': return '⚠';
      case 'severe': return '✗';
      default: return '●';
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'Arial' }}>
        <h1>Transit Analysis Dashboard</h1>
        <div>Loading transit data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', fontFamily: 'Arial' }}>
        <h1>Transit Analysis Dashboard</h1>
        <div style={{ color: 'red', background: '#ffe6e6', padding: '15px', borderRadius: '5px', margin: '20px 0' }}>
          <strong>Connection Error:</strong> {error}
          <br />
          <small>Make sure your Flask backend is running on http://localhost:5000</small>
          <br />
          <button onClick={() => window.location.reload()} style={{ marginTop: '10px', padding: '5px 10px' }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  const OverviewView: React.FC = () => {
    // Just display the data directly from your JSON files
    if (!systemStats) return <div>Loading statistics...</div>;

    console.log('systemStats:', systemStats); // Debug log

    const routeStopsFromStats = systemStats.route_stops;
    const stopAnalysisFromStats = systemStats.stop_analysis;

    // Add safety checks
    if (!routeStopsFromStats || !stopAnalysisFromStats) {
      return (
        <div style={{ padding: '20px' }}>
          <h2>Data Loading Issue</h2>
          <div style={{ background: '#fff3cd', padding: '15px', borderRadius: '5px', border: '1px solid #ffeaa7' }}>
            <p><strong>Debug Info:</strong></p>
            <pre style={{ fontSize: '12px', background: '#f8f9fa', padding: '10px', borderRadius: '3px', overflow: 'auto' }}>
              {JSON.stringify(systemStats, null, 2)}
            </pre>
          </div>
        </div>
      );
    }

    // Use the data directly from your route_stops.json and stop_analysis.json
    const totalRoutes = Object.keys(routeStopsFromStats).length;
    const totalStops = Object.values(routeStopsFromStats).reduce((sum, route) => sum + route.total_stops, 0);

    return (
      <div style={{ padding: '20px' }}>
        <h2>System Overview</h2>
        
        {/* Summary Cards - Data directly from your JSON files */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', margin: '20px 0' }}>
          <div style={{ background: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
            <h3 style={{ color: '#6B7280', fontSize: '14px', margin: '0 0 5px 0' }}>Total Routes</h3>
            <p style={{ color: '#3B82F6', fontSize: '24px', fontWeight: 'bold', margin: '0' }}>{totalRoutes}</p>
          </div>
          <div style={{ background: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
            <h3 style={{ color: '#6B7280', fontSize: '14px', margin: '0 0 5px 0' }}>Total Stops</h3>
            <p style={{ color: '#10B981', fontSize: '24px', fontWeight: 'bold', margin: '0' }}>{totalStops}</p>
          </div>
          <div style={{ background: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
            <h3 style={{ color: '#6B7280', fontSize: '14px', margin: '0 0 5px 0' }}>Analyzed Stops</h3>
            <p style={{ color: '#8B5CF6', fontSize: '24px', fontWeight: 'bold', margin: '0' }}>
              {Object.keys(stopAnalysisFromStats).length}
            </p>
          </div>
          <div style={{ background: 'white', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb', textAlign: 'center' }}>
            <h3 style={{ color: '#6B7280', fontSize: '14px', margin: '0 0 5px 0' }}>Data Sources</h3>
            <p style={{ color: '#F59E0B', fontSize: '24px', fontWeight: 'bold', margin: '0' }}>7</p>
            <p style={{ fontSize: '12px', color: '#6B7280', margin: '5px 0 0 0' }}>JSON Files</p>
          </div>
        </div>

        {/* Routes List - Direct from route_stops.json */}
        <div style={{ background: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <div style={{ padding: '20px', borderBottom: '1px solid #e5e7eb' }}>
            <h3 style={{ margin: '0' }}>Routes Overview</h3>
          </div>
          <div style={{ padding: '20px' }}>
            {Object.values(routeStopsFromStats).map((route) => (
              <div 
                key={route.route_name} 
                style={{ 
                  border: '1px solid #e5e7eb', 
                  borderRadius: '8px', 
                  padding: '15px', 
                  margin: '10px 0',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s'
                }}
                onClick={() => {setSelectedRoute(route.route_name); setSelectedView('route')}}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h4 style={{ margin: '0 0 5px 0' }}>{route.route_name}</h4>
                    <p style={{ color: '#6B7280', fontSize: '14px', margin: '0' }}>{route.total_stops} stops</p>
                  </div>
                  <div style={{ display: 'flex', gap: '15px', fontSize: '14px' }}>
                    <span style={{ color: '#10B981' }}>{route.summary.normal_stops_count} Normal</span>
                    <span style={{ color: '#F59E0B' }}>{route.summary.minor_stops_count} Minor</span>
                    <span style={{ color: '#EF4444' }}>{route.summary.severe_stops_count} Severe</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const RouteView: React.FC = () => {
    // Display route data directly from JSON
    const route = routeStopsData[selectedRoute!];
    if (!route) return <div>Route not found</div>;

    // Get stop analyses for this route
    const routeStops = route.stops.map(stopName => {
      const compositeKey = `${selectedRoute}_${stopName}`;
      return stopAnalysisData[compositeKey] || {
        stop_name: stopName,
        problematic_status: 'Normal',
        problematic_type: 'None',
        stop_type: 'Unknown',
        route_long_name: selectedRoute!,
        description: ''
      };
    });

    return (
      <div style={{ padding: '20px' }}>
        <div style={{ background: 'white', borderRadius: '8px', border: '1px solid #e5e7eb', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2 style={{ margin: '0' }}>{selectedRoute}</h2>
            <button 
              onClick={() => setSelectedView('overview')}
              style={{ color: '#3B82F6', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px' }}
            >
              ← Back to Overview
            </button>
          </div>
          
          {/* Route Stats - Direct from route_stops.json */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px', margin: '20px 0' }}>
            <div style={{ textAlign: 'center', padding: '15px', background: '#ecfdf5', borderRadius: '8px' }}>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#10B981', margin: '0' }}>{route.summary.normal_stops_count}</p>
              <p style={{ fontSize: '14px', color: '#6B7280', margin: '5px 0 0 0' }}>Normal Stops</p>
            </div>
            <div style={{ textAlign: 'center', padding: '15px', background: '#fffbeb', borderRadius: '8px' }}>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#F59E0B', margin: '0' }}>{route.summary.minor_stops_count}</p>
              <p style={{ fontSize: '14px', color: '#6B7280', margin: '5px 0 0 0' }}>Minor Issues</p>
            </div>
            <div style={{ textAlign: 'center', padding: '15px', background: '#fef2f2', borderRadius: '8px' }}>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#EF4444', margin: '0' }}>{route.summary.severe_stops_count}</p>
              <p style={{ fontSize: '14px', color: '#6B7280', margin: '5px 0 0 0' }}>Severe Issues</p>
            </div>
          </div>

          {/* Stops List - Direct from stop_analysis.json */}
          <div>
            <h3>Stops on Route</h3>
            {routeStops.map((stop, index) => (
              <div key={stop.stop_name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px', border: '1px solid #e5e7eb', borderRadius: '8px', margin: '8px 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ background: '#dbeafe', color: '#1d4ed8', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold' }}>
                    {index + 1}
                  </span>
                  <div>
                    <p style={{ fontWeight: 'bold', margin: '0' }}>{stop.stop_name}</p>
                    <p style={{ color: '#6B7280', fontSize: '12px', margin: '0' }}>{stop.stop_type}</p>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px', ...getStatusColor(stop.problematic_status) }}>
                  <span>{getStatusIcon(stop.problematic_status)}</span>
                  <span style={{ fontSize: '14px', fontWeight: 'bold' }}>{stop.problematic_status}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', fontFamily: 'Arial, sans-serif' }}>
      {/* Header */}
      <header style={{ background: 'white', borderBottom: '1px solid #e5e7eb', padding: '15px 0' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '32px', height: '32px', background: '#3B82F6', borderRadius: '4px' }}></div>
            <h1 style={{ margin: '0', fontSize: '20px', color: '#111827' }}>Transit Analysis Dashboard</h1>
          </div>
          <div>
            <input
              type="text"
              placeholder="Search routes or stops..."
              style={{ padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: '6px', fontSize: '14px' }}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {selectedView === 'overview' && <OverviewView />}
        {selectedView === 'route' && <RouteView />}
      </main>
    </div>
  );
};

export default App;