import React, { useEffect, useState, useContext } from 'react';
import { useParams } from 'react-router-dom';

import { useRouteData } from '../contexts/RouteDataContext';
import { useStopData } from '../contexts/StopDataContext';
import { GlobalDataContext } from '../contexts/GlobalDataContext';

import PerformanceSummaryCard from '../cards/PerformanceSummaryCard';
import { StopData, RouteData } from '../contexts/DataInterfaces';
import OverviewCard from '../cards/OverviewCard';
import DirectionBreakdownCard from '../cards/DirectionBreakdownCard';
import RouteBreakdownCard from '../cards/RouteBreakdownCard';

// Type guard for StopData
const isStopData = (data: StopData | RouteData): data is StopData => {
  return (data as StopData).stop_name !== undefined;
};

// Type guard for RouteData
const isRouteData = (data: StopData | RouteData): data is RouteData => {
  return (data as RouteData).route_id !== undefined;
};

const UnifiedLayout: React.FC = () => {
  const { routeId } = useParams<{ routeId: string }>();
  const { parentId } = useParams<{ parentId: string }>();
  
  const { routeData, setRouteId, isLoading: routeLoading, error: routeError } = useRouteData();
  const { stopData, setParentId, isLoading: stopLoading, error: stopError } = useStopData();
  const globalData = useContext(GlobalDataContext);

  const [dataToShow, setDataToShow] = useState<StopData | RouteData | null>(null);

  useEffect(() => {
    if (routeId) {
      setRouteId(routeId);
      setDataToShow(routeData);
    } else if (parentId) {
      setParentId(parentId);
      setDataToShow(stopData);
    }
  }, [routeId, parentId, routeData, stopData, setRouteId, setParentId]);

  // Loading and error states
  if (routeLoading || stopLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-32 bg-gray-200 rounded-lg"></div>
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-24 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (routeError || stopError) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <h3 className="font-medium">Error Loading Data</h3>
          <p className="text-sm mt-1">{routeError || stopError}</p>
        </div>
      </div>
    );
  }

  if (!dataToShow) {
    return (
      <div className="p-6 text-center text-gray-500">No data available</div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Overview Card */}
        <OverviewCard data={dataToShow} />

        {/* Performance Summary */}
        {dataToShow.performance_summary && (
          <PerformanceSummaryCard 
            title="Performance Summary" 
            data={dataToShow.performance_summary} 
          />
        )}

        {/* Route Layout: Direction Breakdown */}
        {isRouteData(dataToShow) && dataToShow.directions && (
          <DirectionBreakdownCard 
            title="Direction Breakdown" 
            directions={dataToShow.directions}
            routeId={dataToShow.route_id}
            globalData={globalData}
          />
        )}

        {/* Stop Layout: Route Breakdown */}
        {isStopData(dataToShow) && dataToShow.routes && (
          <RouteBreakdownCard 
            title="Route Breakdown" 
            routes={dataToShow.routes}
            parentStation={dataToShow.parent_station}
            globalData={globalData}
          />
        )}

      </div>
    </div>
  );
};

export default UnifiedLayout;