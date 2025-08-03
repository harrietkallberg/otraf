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
    setShowRoutes(!showRoutes)
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

  // Calculate summary metrics from the stop_nav data structure
  const totalRoutes = stopData.on_routes?.length || 0
  const totalStopIds = stopData.stop_ids?.length || 0
  
  // Calculate totals from stop_summary
  const totalLabels = stopData.stop_summary?.stop_topology?.labels_by_type?.parent_station + 
                     stopData.stop_summary?.stop_topology?.labels_by_type?.stop_id +
                     stopData.stop_summary?.direction_topology?.labels_by_type?.direction_id +
                     stopData.stop_summary?.direction_topology?.labels_by_type?.stop_id || 0
                     
  const totalViolations = stopData.stop_summary?.stop_topology?.violations_by_type?.parent_station + 
                         stopData.stop_summary?.stop_topology?.violations_by_type?.stop_id +
                         stopData.stop_summary?.direction_topology?.violations_by_type?.direction_id +
                         stopData.stop_summary?.direction_topology?.violations_by_type?.stop_id || 0

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

        {/* Performance Summary */}
        {stopData.performance_summary && (
          <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Overall Performance</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-green-600">
                  {stopData.performance_summary.overall_on_time_rate.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1">On Time</div>
              </div>
              <div className="bg-red-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-red-600">
                  {stopData.performance_summary.overall_too_late_rate.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1">Too Late</div>
              </div>
              <div className="bg-amber-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-amber-600">
                  {stopData.performance_summary.overall_too_early_rate.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1">Too Early</div>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {stopData.performance_summary.average_departure_delay.toFixed(0)}s
                </div>
                <div className="text-sm text-gray-600 mt-1">Avg Delay</div>
              </div>
            </div>
          </div>
        )}

        {/* Summary Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard 
            title="Total Routes" 
            value={totalRoutes}
            subtitle="serving this stop"
            bgColor="bg-blue-50"
            iconColor="text-blue-600"
          />
          <MetricCard 
            title="Stop IDs" 
            value={totalStopIds}
            subtitle="physical locations"
            bgColor="bg-green-50"
            iconColor="text-green-600"
          />
          <MetricCard 
            title="Labels" 
            value={totalLabels}
            subtitle="metadata entries"
            bgColor="bg-purple-50"
            iconColor="text-purple-600"
          />
          <MetricCard 
            title="Violations" 
            value={totalViolations}
            subtitle="issues detected"
            bgColor="bg-red-50"
            iconColor="text-red-600"
          />
        </div>

        {/* Label/Violation Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {stopData.stop_summary && (
            <>
              <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Label Distribution</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Stop Topology - Parent Station</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.stop_topology?.labels_by_type?.parent_station || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Stop Topology - Stop ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.stop_topology?.labels_by_type?.stop_id || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Direction Topology - Direction ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.direction_topology?.labels_by_type?.direction_id || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Direction Topology - Stop ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.direction_topology?.labels_by_type?.stop_id || 0}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Violation Distribution</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Stop Topology - Parent Station</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.stop_topology?.violations_by_type?.parent_station || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Stop Topology - Stop ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.stop_topology?.violations_by_type?.stop_id || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Direction Topology - Direction ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.direction_topology?.violations_by_type?.direction_id || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Direction Topology - Stop ID</span>
                    <span className="text-sm font-medium text-gray-900">
                      {stopData.stop_summary.direction_topology?.violations_by_type?.stop_id || 0}
                    </span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

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

// Route Detail Card - Updated for stop_nav structure
const RouteDetailCard: React.FC<{ 
  route: any
  routeId: string
  isExpanded: boolean
  onToggleExpansion: () => void
}> = ({ route, routeId, isExpanded, onToggleExpansion }) => {
  
  // Calculate totals from route_summary
  const totalLabels = route.route_summary?.stop_topology?.labels_by_type?.parent_station + 
                     route.route_summary?.stop_topology?.labels_by_type?.stop_id +
                     route.route_summary?.direction_topology?.labels_by_type?.direction_id +
                     route.route_summary?.direction_topology?.labels_by_type?.stop_id || 0
                     
  const totalViolations = route.route_summary?.stop_topology?.violations_by_type?.parent_station + 
                         route.route_summary?.stop_topology?.violations_by_type?.stop_id +
                         route.route_summary?.direction_topology?.violations_by_type?.direction_id +
                         route.route_summary?.direction_topology?.violations_by_type?.stop_id || 0

  return (
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
              {/* Performance Summary */}
              {route.performace_summary && (
                <div className="mt-2 flex items-center space-x-4 text-xs text-gray-600">
                  <span className="text-green-600">
                    On Time: {route.performace_summary.overall_on_time_rate.toFixed(1)}%
                  </span>
                  <span className="text-red-600">
                    Late: {route.performace_summary.overall_too_late_rate.toFixed(1)}%
                  </span>
                  <span className="text-amber-600">
                    Early: {route.performace_summary.overall_too_early_rate.toFixed(1)}%
                  </span>
                </div>
              )}
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
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border bg-blue-100 text-blue-800 border-blue-200">
              {totalLabels} Labels
            </span>
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${
              totalViolations > 0 
                ? 'bg-red-100 text-red-800 border-red-200' 
                : 'bg-green-100 text-green-800 border-green-200'
            }`}>
              {totalViolations} Violations
            </span>
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
}

// Direction Section - Updated for stop_nav structure
const DirectionSection: React.FC<{ 
  directionId: string
  direction: any
}> = ({ directionId, direction }) => {
  
  // Get stops from canonical_position
  const stops = direction.stops_in_direction?.canonical_position || {}
  const stopCount = Object.keys(stops).length
  
  // Calculate totals from direction_summary
  const totalLabels = direction.direction_summary?.direction_topology?.labels_by_type?.direction_id || 0
                     
  const totalViolations = direction.direction_summary?.direction_topology?.violations_by_type?.direction_id  || 0

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <h5 className="font-medium text-gray-800">Direction {directionId}</h5>
          <span className="text-xs text-gray-500">{stopCount} stops</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            {totalLabels} labels
          </span>
          {totalViolations > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
              {totalViolations} violations
            </span>
          )}
        </div>
      </div>
      
      <div className="space-y-2">
        {Object.entries(stops)
          .sort(([a], [b]) => parseInt(a) - parseInt(b))
          .map(([position, stop]: [string, any]) => (
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
                    <p className="text-xs text-gray-500">Parent: {stop.parent_station}</p>
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
                {/* Show regulatory stop indicator */}
                {stop.stop_id_summary?.regulatory_stops?.regulatory_stop_ids > 0 && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                    REG
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default StopLayout