import React, { useState } from 'react';
import { RouteData } from '../contexts/DataInterfaces';
import CompactHistogram from './CompactHistogram';

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
  const [selectedTimeType, setSelectedTimeType] = useState<string>('scheduled');

  // Get available time types from globalData
  const availableTimeTypes = globalData?.time_types || ['scheduled'];

  // Find stops in this direction that belong to our parent station
  const relevantStops = direction.stop_ids_in_direction 
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
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Time Type:</label>
          <select
            value={selectedTimeType}
            onChange={(e) => setSelectedTimeType(e.target.value)}
            className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            {availableTimeTypes.map((timeType: string) => (
              <option key={timeType} value={timeType}>
                {timeType.replace('_', ' ').toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Always show relevant stops for this direction */}
      {relevantStops.length > 0 && (
        <div className="space-y-3">
          {relevantStops.map((stopIdData) => (
            <StopDetailsCard 
              key={stopIdData.stop_id}
              stopIdData={stopIdData}
              routeId={routeId}
              directionId={directionId}
              globalData={globalData}
              selectedTimeType={selectedTimeType}
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

// Reusable stop details card component
interface StopDetailsCardProps {
  stopIdData: any;
  routeId: string;
  directionId: string;
  globalData: any;
  selectedTimeType: string;
}

const StopDetailsCard: React.FC<StopDetailsCardProps> = ({ 
  stopIdData, 
  routeId, 
  directionId, 
  globalData,
  selectedTimeType
}) => {
  const [showDetails, setShowDetails] = useState(false);
  
  const isRegulatory = stopIdData.stop_id_label_keys?.some((key: string) => 
    key.includes('regulatory_stops')
  );

  const labelCount = stopIdData.stop_id_label_keys?.length || 0;
  const violationCount = stopIdData.stop_id_violation_keys?.length || 0;
  const performanceCount = stopIdData.stop_id_performance_keys?.length || 0;

  // Get the actual data for the selected time type
  const getDataForTimeType = (timeType: string) => {
    // Get labels (these don't change by time type)
    const labels = (stopIdData.stop_id_label_keys || [])
      .map((key: string) => globalData?.labels?.[key])
      .filter(Boolean);
    
    // Get violations (these don't change by time type)
    const violations = (stopIdData.stop_id_violation_keys || [])
      .map((key: string) => globalData?.violations?.[key])
      .filter(Boolean);
    
    // Find performance data for this specific time type
    const performanceKeys = stopIdData.stop_id_performance_keys || [];
    const performanceKey = performanceKeys.find((key: string) => 
      key.includes(`_${timeType}_`) || key.includes(timeType)
    ) || performanceKeys[0]; // Fallback to first available
    
    const performanceData = performanceKey ? globalData?.performance?.[performanceKey] : null;

    return {
      labels,
      violations,
      performanceData
    };
  };

  const currentData = getDataForTimeType(selectedTimeType);

  return (
    <div className="border border-gray-200 rounded p-3 bg-white">
      {/* Stop Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
            isRegulatory 
              ? 'bg-amber-200 text-amber-800' 
              : 'bg-blue-100 text-blue-800'
          }`}>
            {stopIdData.position}
          </div>
          <div>
            <h5 className="font-medium text-gray-900">{stopIdData.stop_name}</h5>
            <p className="text-xs text-gray-600">Stop ID: {stopIdData.stop_id}</p>
          </div>
        </div>
        
        {/* Stop-level markers and individual toggle */}
        <div className="flex items-center space-x-2">
          {isRegulatory && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-amber-200 text-amber-800">
              Regulatory
            </span>
          )}
          {labelCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              {labelCount} labels
            </span>
          )}
          {violationCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
              {violationCount} violations
            </span>
          )}
          {performanceCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              {performanceCount} analytics
            </span>
          )}
          
          {/* Individual Show/Hide toggle on each stop */}
          {(labelCount > 0 || violationCount > 0 || performanceCount > 0) && (
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-blue-700 hover:text-blue-800"
            >
              {showDetails ? 'Hide' : 'Show'}
            </button>
          )}
        </div>
      </div>

      {/* Show details when expanded */}
      {showDetails && (
        <div className="border-t border-gray-200 pt-3">
          <StopDetailsView 
            labels={currentData.labels}
            violations={currentData.violations}
            performanceData={currentData.performanceData}
          />
        </div>
      )}
    </div>
  );
};

// Component that shows the detailed data for a stop (reused from StopSequence)
const StopDetailsView: React.FC<{
  labels: any[];
  violations: any[];
  performanceData: any;
}> = ({ labels, violations, performanceData }) => {
  
  return (
    <div className="space-y-4">
      
      {/* Performance Details */}
      {performanceData?.analytics && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Performance Analysis</h4>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Punctuality Breakdown */}
            <div className="bg-gray-50 rounded-lg p-3">
              <h5 className="text-xs font-medium text-gray-600 mb-2">PUNCTUALITY DISTRIBUTION</h5>
              <div className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-green-600">On Time</span>
                  <span className="text-xs font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-yellow-600">Too Early</span>
                  <span className="text-xs font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.too_early}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-red-600">Too Late</span>
                  <span className="text-xs font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.too_late}%</span>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-50 rounded-lg p-3">
              <h5 className="text-xs font-medium text-gray-600 mb-2">DELAY STATISTICS</h5>
              <div className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-600">Mean</span>
                  <span className="text-xs font-medium">{performanceData.analytics.punctuality.basic_statistics.mean_delay.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-600">Median</span>
                  <span className="text-xs font-medium">{performanceData.analytics.punctuality.basic_statistics.median_delay}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-600">Sample Size</span>
                  <span className="text-xs font-medium">{performanceData.analytics.punctuality.sample_size.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Compact Histogram */}
            <CompactHistogram performanceData={performanceData} />
          </div>
        </div>
      )}

      {/* Labels */}
      {labels && labels.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">Labels ({labels.length})</h4>
          <div className="space-y-2">
            {labels.map((label, index) => (
              <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-2">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {label.label_type}
                      </span>
                    </div>
                    <p className="text-xs text-gray-700">{label.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Violations */}
      {violations && violations.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">Violations ({violations.length})</h4>
          <div className="space-y-2">
            {violations.map((violation, index) => {
              const getSeverityColor = (severity: number) => {
                if (severity >= 5) return 'bg-red-100 text-red-800 border-red-200'
                if (severity >= 3) return 'bg-orange-100 text-orange-800 border-orange-200'
                return 'bg-yellow-100 text-yellow-800 border-yellow-200'
              }

              return (
                <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-2">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          {violation.violation_type}
                        </span>
                        {violation.severity && (
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(violation.severity)}`}>
                            Severity {violation.severity}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-700">{violation.description}</p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* No Additional Data */}
      {(!labels || labels.length === 0) && (!violations || violations.length === 0) && !performanceData?.analytics && (
        <div className="text-center py-4 text-gray-500 text-xs">
          No detailed data found for this combination
        </div>
      )}
    </div>
  )
};

export default RouteBreakdownCard;