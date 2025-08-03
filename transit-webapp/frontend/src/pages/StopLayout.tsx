import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useStopData } from '../contexts/StopDataContext'

const StopLayout: React.FC = () => {
  const { parentId } = useParams<{ parentId: string }>()
  const { stopData, setParentId, isLoading, error } = useStopData()
  const [expandedRoutes, setExpandedRoutes] = useState<Set<string>>(new Set())
  const [showRoutes, setShowRoutes] = useState(false)

  const toggleRouteExpansion = (routeId: string) => {
    setExpandedRoutes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(routeId)) {
        newSet.delete(routeId)
      } else {
        newSet.add(routeId)
      }
      return newSet
    })
  }

  const toggleRoutesVisibility = () => {
    setShowRoutes(!showRoutes); // Toggle visibility of the whole route details section
  }

  useEffect(() => {
    if (parentId) {
      console.log('StopLayout setting parentId:', parentId)
      setParentId(parentId)
    }
  }, [parentId, setParentId])

  if (!parentId) {
    return <div className="p-6 text-center text-gray-500">Parent ID is missing</div>
  }

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-32 bg-gray-200 rounded-lg"></div>
          <div className="grid grid-cols-4 gap-4">
            {[1,2,3,4].map(i => <div key={i} className="h-24 bg-gray-200 rounded-lg"></div>)}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <h3 className="font-medium">Error Loading Stop Data</h3>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    )
  }

  if (!stopData) {
    return (
      <div className="p-6 text-center">
        <div className="text-gray-400 text-lg">No stop data available</div>
      </div>
    )
  }

  const dataMatches = stopData && (
    stopData.parent_station === parentId || 
    stopData.parent_id === parentId ||
    stopData.id === parentId ||
    stopData.stop_id === parentId
  )

  if (!dataMatches) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-32 bg-gray-200 rounded-lg"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Header Section */}
        <div className="bg-white shadow-sm rounded-lg border border-gray-200">
          <div className="px-6 py-8">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">{stopData.stop_name}</h1>
                    <p className="text-sm text-gray-500 mt-1">Parent Station: {stopData.parent_station}</p>
                  </div>
                </div>
                
                <div className="mt-6">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Stop IDs</h3>
                  <div className="flex flex-wrap gap-2">
                    {stopData.stop_ids?.map((id: string, index: number) => (
                      <span key={index} className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 border">
                        {id}
                      </span>
                    ))}
                  </div>
                </div>

                {stopData.on_routes && (
                  <div className="mt-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Routes Serving This Stop</h3>
                    <div className="flex flex-wrap gap-2">
                      {stopData.on_routes.map((routeId: string, index: number) => (
                        <span key={index} className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200">
                          {routeId}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Summary Metrics - Only using data from stop_summary */}
        {stopData.stop_summary && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard 
              title="Total Routes" 
              value={stopData.stop_summary.total_routes}
              subtitle="serving this stop"
              bgColor="bg-blue-50"
              iconColor="text-blue-600"
            />
            <MetricCard 
              title="Stop IDs" 
              value={stopData.stop_summary.total_stop_ids}
              subtitle="physical locations"
              bgColor="bg-green-50"
              iconColor="text-green-600"
            />
            <MetricCard 
              title="Labels" 
              value={stopData.stop_summary.total_labels}
              subtitle="metadata entries"
              bgColor="bg-purple-50"
              iconColor="text-purple-600"
            />
            <MetricCard 
              title="Violations" 
              value={stopData.stop_summary.total_violations}
              subtitle="issues detected"
              bgColor="bg-red-50"
              iconColor="text-red-600"
            />
          </div>
        )}

        {/* Label/Violation Breakdown */}
        {stopData.stop_summary && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {stopData.stop_summary.label_counts_by_type && (
              <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Label Distribution</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Parent Station</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.label_counts_by_type.parent_station}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Stop ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.label_counts_by_type.stop_id}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {stopData.stop_summary.violation_counts_by_type && (
              <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Violation Distribution</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Parent Station</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.violation_counts_by_type.parent_station}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Stop ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.violation_counts_by_type.stop_id}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Routes Detail Section */}
        {stopData.routes && Object.keys(stopData.routes).length > 0 && (
          <div className="bg-white shadow-sm rounded-lg border border-gray-200">
            {/* Header with Show Routes Button */}
            <div className="px-6 py-4 flex justify-between items-center border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Route Breakdown</h2>
              <button
                onClick={toggleRoutesVisibility}
                className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
              >
                {showRoutes ? "Hide Routes" : "Show Routes"}
              </button>
            </div>

            {/* Conditionally render routes */}
            {showRoutes && (
              <div className="p-6 space-y-6">
                {Object.entries(stopData.routes).map(([routeId, route]: [string, any]) => (
                  <RouteDetailCard 
                    key={routeId} 
                    route={route} 
                    routeId={routeId}
                    isExpanded={expandedRoutes.has(routeId)}
                    onToggleExpansion={() => toggleRouteExpansion(routeId)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  )
}

// Metric Card Component
const MetricCard: React.FC<{
  title: string
  value: number
  subtitle: string
  bgColor: string
  iconColor: string
}> = ({ title, value, subtitle, bgColor, iconColor }) => (
  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
    <div className="flex items-center">
      <div className={`flex-shrink-0 ${bgColor} rounded-lg p-3`}>
        <svg className={`w-6 h-6 ${iconColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2h-2a2 2 0 00-2 2z" />
        </svg>
      </div>
      <div className="ml-4 flex-1">
        <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">{title}</p>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
      </div>
    </div>
  </div>
)

// Route Detail Card
const RouteDetailCard: React.FC<{ 
  route: any
  routeId: string
  isExpanded: boolean
  onToggleExpansion: () => void
}> = ({ route, routeId, isExpanded, onToggleExpansion }) => (
  <div className="border border-gray-200 rounded-lg overflow-hidden">
    <div className="bg-gray-50 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <span className="text-sm font-bold text-blue-700">{route.route_short_name}</span>
            </div>
          </div>
          <div className="flex-1">
            <h3 className="text-base font-semibold text-gray-900">{route.route_long_name}</h3>
            <p className="text-sm text-gray-500">Route ID: {route.route_id}</p>
            <button
              onClick={onToggleExpansion}
              className="inline-flex items-center mt-2 text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
            >
              {isExpanded ? (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7" />
                  </svg>
                  Hide Directions
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                  </svg>
                  Show Directions
                </>
              )}
            </button>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {route.total_labels_on_route && (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border bg-blue-100 text-blue-800 border-blue-200">
              {route.total_labels_on_route.total_labels} Labels
            </span>
          )}
          {route.total_violations_on_route && (
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${
              route.total_violations_on_route.total_violations > 0 
                ? 'bg-red-100 text-red-800 border-red-200' 
                : 'bg-green-100 text-green-800 border-green-200'
            }`}>
              {route.total_violations_on_route.total_violations} Violations
            </span>
          )}
        </div>
      </div>
    </div>

    {isExpanded && route.directions && (
      <div className="px-6 py-4 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-4">Directions</h4>
        <div className="space-y-4">
          {Object.entries(route.directions).map(([directionId, direction]: [string, any]) => (
            <DirectionSection 
              key={directionId} 
              directionId={directionId} 
              direction={direction}
            />
          ))}
        </div>
      </div>
    )}
  </div>
)

// Direction Section
const DirectionSection: React.FC<{ 
  directionId: string
  direction: any
}> = ({ directionId, direction }) => (
  <div className="bg-gray-50 rounded-lg p-4">
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center space-x-3">
        <h5 className="font-medium text-gray-800">Direction {directionId}</h5>
        <span className="text-xs text-gray-500">
          {direction.stops_in_direction?.canonical_position ? 
            Object.keys(direction.stops_in_direction.canonical_position).length : 0} stops
        </span>
      </div>
      <div className="flex items-center space-x-2">
        {direction.direction_label_keys?.length > 0 && (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            {direction.direction_label_keys.length} labels
          </span>
        )}
        {direction.direction_violation_keys?.length > 0 && (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
            {direction.direction_violation_keys.length} violations
          </span>
        )}
      </div>
    </div>
    
    {direction.stops_in_direction?.canonical_position && (
      <div className="space-y-2">
        {Object.entries(direction.stops_in_direction.canonical_position).map(([position, stop]: [string, any]) => (
          <div key={position} className="bg-white rounded-md border border-gray-200 p-3">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-6 h-6 bg-gray-100 rounded-full flex items-center justify-center">
                      <span className="text-xs font-medium text-gray-600">{position}</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{stop.stop_name}</p>
                    <p className="text-xs text-gray-500">Stop ID: {stop.stop_id}</p>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {stop.stop_id_performance_keys?.length > 0 && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    {stop.stop_id_performance_keys.length} metrics
                  </span>
                )}
                {stop.stop_id_label_keys?.length > 0 && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {stop.stop_id_label_keys.length} labels
                  </span>
                )}
                {stop.stop_id_violation_keys?.length > 0 && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    {stop.stop_id_violation_keys.length} violations
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
)

export default StopLayout