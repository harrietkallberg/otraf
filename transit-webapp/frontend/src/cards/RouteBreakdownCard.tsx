// src/components/cards/RouteBreakdownCard.tsx
import React, { useState } from 'react';
import { RouteData, StopIdDataWithPosition } from '../shared/types';
import { StopCard, TimeTypeSelector} from '../components/shared';

// Correct interface for the main component
interface RouteBreakdownCardProps {
  title: string;
  routes: { [routeId: string]: RouteData };
  parentStation: string;
  globalData: any;
}

// Main component - this is what gets imported in UnifiedLayout
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

// Internal component interfaces
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

// Internal direction component
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
  const [selectedTimeType, setSelectedTimeType] = useState<string>('scheduled');

  // Get available time types from globalData
  const availableTimeTypes = globalData?.time_types || ['scheduled'];

  // Find stops in this direction that belong to our parent station
  const relevantStops: StopIdDataWithPosition[] = direction.stop_ids_in_direction 
    ? Object.entries(direction.stop_ids_in_direction)
        .filter(([_, stopIdData]: [string, any]) => stopIdData.parent_station === parentStation)
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .map(([position, stopIdData]: [string, any]) => ({
          position: parseInt(position),
          ...stopIdData
        }))
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
            <span className="text-xs text-gray-600">
              {relevantStops.length} stop{relevantStops.length !== 1 ? 's' : ''}
            </span>
          </div>
        </div>

        {/* Time Type Selector - always visible when direction is shown */}
        <TimeTypeSelector
          selectedTimeType={selectedTimeType}
          availableTimeTypes={availableTimeTypes}
          onTimeTypeChange={setSelectedTimeType}
          size="sm"
        />
      </div>

      {/* Always show relevant stops for this direction */}
      {relevantStops.length > 0 && (
        <div className="space-y-3">
          {relevantStops.map((stopIdData) => (
            <StopCard 
              key={stopIdData.stop_id}
              stopIdData={stopIdData}
              globalData={globalData}
              externalTimeType={selectedTimeType}
              size="sm"
              showTimeSelector={false}
              positionBadgeSize="sm"
            />
          ))}
        </div>
      )}

      {relevantStops.length === 0 && (
        <div className="text-sm text-gray-500 text-center py-3">
          No stops found for this direction at this station
        </div>
      )}
    </div>
  );
};

// Make sure to export the main component as default
export default RouteBreakdownCard;