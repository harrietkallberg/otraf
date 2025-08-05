import React, { useContext } from 'react';
import { StopData, RouteData } from '../shared/types';
import { GlobalDataContext } from '../contexts/GlobalDataContext';

// Type guard to check if the data is of type StopData
const isStopData = (data: StopData | RouteData): data is StopData => {
  return (data as StopData).stop_name !== undefined;
};

// Type guard to check if the data is of type RouteData
const isRouteData = (data: StopData | RouteData): data is RouteData => {
  return (data as RouteData).route_id !== undefined;
};

interface OverviewCardProps {
  data: StopData | RouteData;
}

const OverviewCard: React.FC<OverviewCardProps> = ({ data }) => {
  const globalData = useContext(GlobalDataContext);

  return (
    <div className="bg-white shadow-sm rounded-lg border border-gray-200">
      <div className="px-6 py-8">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            
            {/* Route Overview */}
            {isRouteData(data) && (
              <>
                <div className="flex items-center space-x-3 mb-6">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-sky-100 rounded-lg flex items-center justify-center">
                      <span className="text-lg font-bold text-sky-600">
                        {data.route_short_name}
                      </span>
                    </div>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">Route {data.route_short_name}</h1>
                    <p className="text-lg text-gray-600 mt-1">{data.route_long_name}</p>
                    <p className="text-sm text-gray-500 mt-1">Route ID: {data.route_id}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Route Coverage</h3>
                    <div className="text-2xl font-bold text-gray-900">{data.on_stops?.length || 0}</div>
                    <div className="text-xs text-gray-500">Parent stations served</div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Directions</h3>
                    <div className="text-2xl font-bold text-gray-900">
                      {data.directions ? Object.keys(data.directions).length : 0}
                    </div>
                    <div className="text-xs text-gray-500">Operating directions</div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Trip Instances</h3>
                    <div className="text-2xl font-bold text-gray-900">
                      {data.route_summary?.total_trip_instances?.toLocaleString() || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500">Total recorded trips</div>
                  </div>
                </div>

                {data.on_stops && data.on_stops.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-sm font-medium text-gray-700 mb-3">Stops on Route</h3>
                    <div className="flex flex-wrap gap-2">
                      {data.on_stops.map((parentStationId: string, index: number) => (
                        <span key={index} className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-orange-100 text-orange-800 border border-orange-200">
                          {globalData?.stops?.[parentStationId]?.stop_name || parentStationId}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Stop Overview */}
            {isStopData(data) && (
              <>
                <div className="flex items-center space-x-3 mb-6">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">{data.stop_name}</h1>
                    <p className="text-sm text-gray-500 mt-1">Parent Station: {data.parent_station}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Physical Stops</h3>
                    <div className="text-2xl font-bold text-gray-900">{data.stop_ids?.length || 0}</div>
                    <div className="text-xs text-gray-500">Stop locations</div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Routes Served</h3>
                    <div className="text-2xl font-bold text-gray-900">{data.on_routes?.length || 0}</div>
                    <div className="text-xs text-gray-500">Bus routes</div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Trip Instances</h3>
                    <div className="text-2xl font-bold text-gray-900">
                      {data.stop_summary?.total_trip_instances?.toLocaleString() || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500">Total recorded trips</div>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Stop IDs */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-3">Stop IDs</h3>
                    <div className="flex flex-wrap gap-2">
                      {data.stop_ids?.map((id: string, index: number) => (
                        <span key={index} className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 border">
                          {id}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Routes */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-3">Routes Serving This Stop</h3>
                    <div className="flex flex-wrap gap-2">
                      {data.on_routes?.map((routeId: string, index: number) => (
                        <div key={index} className="w-8 h-8 bg-sky-100 rounded-lg flex items-center justify-center border border-sky-200">
                          <span className="text-xs font-bold text-sky-600">
                            {globalData?.routes?.[routeId]?.route_short_name || routeId}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}

          </div>
        </div>

        {/* Summary Statistics */}
        {(isRouteData(data) ? data.route_summary : data.stop_summary) && (
          <div className="mt-8 pt-6 border-t border-gray-200">
            <h3 className="text-sm font-medium text-gray-700 mb-4">Data Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <div className="text-lg font-bold text-blue-600">
                  {isRouteData(data) 
                    ? (data.route_summary?.stop_topology?.labels_by_type?.parent_station || 0) + 
                      (data.route_summary?.direction_topology?.labels_by_type?.direction_id || 0)
                    : (data.stop_summary?.stop_topology?.labels_by_type?.parent_station || 0) + 
                      (data.stop_summary?.direction_topology?.labels_by_type?.direction_id || 0)
                  }
                </div>
                <div className="text-xs text-blue-800">Total Labels</div>
              </div>
              
              <div className="text-center p-3 bg-red-50 rounded-lg">
                <div className="text-lg font-bold text-red-600">
                  {isRouteData(data) 
                    ? (data.route_summary?.stop_topology?.violations_by_type?.parent_station || 0) + 
                      (data.route_summary?.direction_topology?.violations_by_type?.direction_id || 0)
                    : (data.stop_summary?.stop_topology?.violations_by_type?.parent_station || 0) + 
                      (data.stop_summary?.direction_topology?.violations_by_type?.direction_id || 0)
                  }
                </div>
                <div className="text-xs text-red-800">Total Violations</div>
              </div>
              
              <div className="text-center p-3 bg-amber-50 rounded-lg">
                <div className="text-lg font-bold text-amber-600">
                  {isRouteData(data) 
                    ? data.route_summary?.performance?.available_performace_analytics || 0
                    : data.stop_summary?.performance?.available_performace_analytics || 0
                  }
                </div>
                <div className="text-xs text-amber-800">Analytics Available</div>
              </div>
              
              <div className="text-center p-3 bg-orange-50 rounded-lg">
                <div className="text-lg font-bold text-orange-600">
                  {isRouteData(data) 
                    ? data.route_summary?.regulatory_stops?.regulatory_stop_ids || 0
                    : data.stop_summary?.regulatory_stops?.regulatory_stop_ids || 0
                  }
                </div>
                <div className="text-xs text-orange-800">Regulatory Stops</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OverviewCard;