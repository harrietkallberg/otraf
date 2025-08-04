import React, { useState } from 'react';
import { RouteData } from '../contexts/DataInterfaces';

interface RouteBreakdownCardProps {
  title: string;
  routes: { [routeId: string]: RouteData };
  parentStation: string;
  globalData: any;
}

const RouteBreakdownCard: React.FC<RouteBreakdownCardProps> = ({ 
  title, 
  routes, 
  parentStation, 
  globalData 
}) => {
  const [showRoutes, setShowRoutes] = useState(false);

  const toggleRoutesVisibility = () => {
    setShowRoutes(!showRoutes);
  };

  return (
    <div className="bg-white shadow-sm rounded-lg border border-gray-200">
      <div className="px-6 py-4 flex justify-between items-center border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        <button
          onClick={toggleRoutesVisibility}
          className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
        >
          {showRoutes ? "Hide Routes" : "Show Routes"}
        </button>
      </div>

      {showRoutes && (
        <div className="p-6 space-y-6">
          {Object.entries(routes).map(([routeId, route]) => (
            <RouteTile 
              key={routeId}
              routeId={routeId}
              route={route}
              parentStation={parentStation}
              globalData={globalData}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface RouteTileProps {
  routeId: string;
  route: RouteData;
  parentStation: string;
  globalData: any;
}

const RouteTile: React.FC<RouteTileProps> = ({ 
  routeId, 
  route, 
  parentStation, 
  globalData 
}) => {
  const [showDirections, setShowDirections] = useState(false);

  // Calculate markers from route_summary  
  const routeLabels = route.route_summary?.stop_topology?.labels_by_type?.parent_station || 0;
  const routeViolations = route.route_summary?.stop_topology?.violations_by_type?.parent_station || 0;
  const analytics = route.route_summary?.performance?.available_performace_analytics || 0;

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      {/* Route Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-medium text-gray-900">
            Route {route.route_short_name} - {route.route_long_name}
          </h3>
          
          {/* Markers */}
          <div className="flex items-center space-x-3">
            {routeLabels > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {routeLabels} labels
              </span>
            )}
            {routeViolations > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                {routeViolations} violations
              </span>
            )}
            {analytics > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                {analytics} analytics
              </span>
            )}
          </div>
        </div>

        <button
          onClick={() => setShowDirections(!showDirections)}
          className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
        >
          {showDirections ? "Hide Directions" : "Show Directions"}
        </button>
      </div>

      {/* Route Performance Summary */}
      {route.performance_summary && (
        <div className="grid grid-cols-4 gap-4 mb-4 p-3 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">On Time</div>
            <div className="text-lg font-bold text-green-600">
              {route.performance_summary.overall_on_time_rate.toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">Too Early</div>
            <div className="text-lg font-bold text-yellow-600">
              {route.performance_summary.overall_too_early_rate.toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">Too Late</div>
            <div className="text-lg font-bold text-red-600">
              {route.performance_summary.overall_too_late_rate.toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-gray-700">Avg Delay</div>
            <div className="text-lg font-bold text-gray-700">
              {route.performance_summary.average_departure_delay.toFixed(0)}s
            </div>
          </div>
        </div>
      )}

      {/* Directions for this route at this stop */}
      {showDirections && route.directions && (
        <div className="space-y-4">
          {Object.entries(route.directions).map(([directionId, direction]) => (
            <RouteDirectionTile 
              key={directionId}
              routeId={routeId}
              directionId={directionId}
              direction={direction}
              parentStation={parentStation}
              globalData={globalData}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface RouteDirectionTileProps {
  routeId: string;
  directionId: string;
  direction: any; // DirectionData from the route context
  parentStation: string;
  globalData: any;
}

const RouteDirectionTile: React.FC<RouteDirectionTileProps> = ({ 
  routeId, 
  directionId, 
  direction, 
  parentStation, 
  globalData 
}) => {
  const [showStopDetails, setShowStopDetails] = useState(false);
  const [selectedTimeType, setSelectedTimeType] = useState('day');

  // Find stops in this direction that belong to our parent station
  const relevantStops = direction.stop_ids_in_direction 
    ? Object.entries(direction.stop_ids_in_direction)
        .filter(([_, stopData]: [string, any]) => stopData.parent_station === parentStation)
    : [];

  // Calculate markers for this direction at this stop
  const directionLabels = direction.direction_summary?.direction_topology?.labels_by_type?.direction_id || 0;
  const directionViolations = direction.direction_summary?.direction_topology?.violations_by_type?.direction_id || 0;

  return (
    <div className="border border-gray-100 rounded-lg p-3 bg-gray-50">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <h4 className="text-md font-medium text-gray-800">
            Direction {directionId}
          </h4>
          
          {/* Direction Markers */}
          <div className="flex items-center space-x-2">
            {directionLabels > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {directionLabels} labels
              </span>
            )}
            {directionViolations > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                {directionViolations} violations
              </span>
            )}
          </div>
        </div>

        <button
          onClick={() => setShowStopDetails(!showStopDetails)}
          className="text-sm text-blue-600 hover:text-blue-700"
        >
          {showStopDetails ? "Hide Details" : "Show Details"}
        </button>
      </div>

      {/* Show relevant stops for this direction */}
      {showStopDetails && relevantStops.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center space-x-4 mb-3">
            <label className="text-sm font-medium text-gray-700">Time Type:</label>
            <select
              value={selectedTimeType}
              onChange={(e) => setSelectedTimeType(e.target.value)}
              className="text-sm border border-gray-300 rounded-md px-2 py-1"
            >
              <option value="am_rush">AM Rush</option>
              <option value="day">Day</option>
              <option value="pm_rush">PM Rush</option>
              <option value="night">Night</option>
              <option value="weekend">Weekend</option>
            </select>
          </div>

          {relevantStops.map(([position, stopData]: [string, any]) => {
            // Find performance data for this stop
            const performanceKey = `performance_${routeId}_direction_id_stop_id_time_type_${directionId}_${stopData.stop_id}_${selectedTimeType}`;
            const performanceData = globalData?.performance?.[performanceKey];

            return (
              <div key={position} className="border border-gray-200 rounded p-3 bg-white">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h5 className="font-medium text-gray-900">{stopData.stop_name}</h5>
                    <p className="text-sm text-gray-600">
                      Position {position} • Stop ID: {stopData.stop_id}
                    </p>
                  </div>
                  
                  {/* Stop-level markers */}
                  <div className="flex items-center space-x-2">
                    {stopData.stop_id_label_keys?.length > 0 && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {stopData.stop_id_label_keys.length} labels
                      </span>
                    )}
                    {stopData.stop_id_violation_keys?.length > 0 && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        {stopData.stop_id_violation_keys.length} violations
                      </span>
                    )}
                    {performanceData && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        Performance
                      </span>
                    )}
                  </div>
                </div>

                {/* Performance Summary for this stop */}
                {performanceData?.analytics?.punctuality && (
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div className="text-center">
                      <div className="text-gray-600">On Time</div>
                      <div className="font-bold text-green-600">
                        {performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-600">Too Early</div>
                      <div className="font-bold text-yellow-600">
                        {performanceData.analytics.punctuality.punctuality_distribution.percentages.too_early}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-600">Too Late</div>
                      <div className="font-bold text-red-600">
                        {performanceData.analytics.punctuality.punctuality_distribution.percentages.too_late}%
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {relevantStops.length === 0 && showStopDetails && (
        <div className="text-sm text-gray-500 text-center py-3">
          No stops found for this direction at this station
        </div>
      )}
    </div>
  );
};

export default RouteBreakdownCard;