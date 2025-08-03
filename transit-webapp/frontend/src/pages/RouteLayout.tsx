import React, { useEffect, useState, useContext } from 'react'
import { useParams } from 'react-router-dom'
import { useRouteData } from '../contexts/RouteDataContext'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import DirectionDetailCard from '../cards/DirectionDetailCard'
import MetricCard from '../cards/MetricCard'

const RouteLayout: React.FC = () => {
  const { routeId } = useParams<{ routeId: string }>()
  const { routeData, setRouteId, isLoading, error } = useRouteData()
  const globalData = useContext(GlobalDataContext)
  const [expandedDirections, setExpandedDirections] = useState<Set<string>>(new Set())
  const [showDirections, setShowDirections] = useState(false)
  const [showPunctualityFlow, setShowPunctualityFlow] = useState<{[key: string]: boolean}>({})
  const [selectedTimeType, setSelectedTimeType] = useState<{[key: string]: string}>({})
  const [selectedMetric, setSelectedMetric] = useState<{[key: string]: string}>({})

  const toggleDirectionExpansion = (directionId: string) => {
    setExpandedDirections(prev => {
      const newSet = new Set(prev)
      if (newSet.has(directionId)) {
        newSet.delete(directionId)
      } else {
        newSet.add(directionId)
      }
      return newSet
    })
  }

  const toggleDirectionsVisibility = () => {
    setShowDirections(!showDirections)
  }

  const togglePunctualityFlow = (directionId: string) => {
    setShowPunctualityFlow(prev => ({
      ...prev,
      [directionId]: !prev[directionId]
    }))
    
    // Set default values if not already set
    if (!selectedTimeType[directionId]) {
      setSelectedTimeType(prev => ({
        ...prev,
        [directionId]: 'day'
      }))
    }
    if (!selectedMetric[directionId]) {
      setSelectedMetric(prev => ({
        ...prev,
        [directionId]: 'on_time'
      }))
    }
  }

  const updateTimeType = (directionId: string, timeType: string) => {
    setSelectedTimeType(prev => ({
      ...prev,
      [directionId]: timeType
    }))
  }

  const updateMetric = (directionId: string, metric: string) => {
    setSelectedMetric(prev => ({
      ...prev,
      [directionId]: metric
    }))
  }

  useEffect(() => {
    if (routeId) {
      console.log('RouteLayout setting routeId:', routeId)
      setRouteId(routeId)
    }
  }, [routeId, setRouteId])

  if (!routeId) {
    return <div className="p-6 text-center text-gray-500">Route ID is missing</div>
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
          <h3 className="font-medium">Error Loading Route Data</h3>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    )
  }

  if (!routeData) {
    return (
      <div className="p-6 text-center">
        <div className="text-gray-400 text-lg">No route data available</div>
      </div>
    )
  }

  const dataMatches = routeData && (
    routeData.route_id === routeId || 
    routeData.id === routeId ||
    String(routeData.route_id) === routeId ||
    String(routeData.id) === routeId
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
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">Route {routeData.route_short_name}</h1>
                    <p className="text-sm text-gray-500 mt-1">{routeData.route_long_name}</p>
                  </div>
                </div>
                
                <div className="mt-6">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Route Details</h3>
                  <div className="flex flex-wrap gap-2">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 border">
                      Route ID: {routeData.route_id}
                    </span>
                  </div>
                </div>

                {routeData.route_summary?.performance && (
                  <div className="mt-4">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Performance Overview</h3>
                    <div className="flex flex-wrap gap-2">
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 border border-green-200">
                        On Time: {routeData.route_summary.performance.overall_on_time_rate}%
                      </span>
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200">
                        Avg Delay: {Math.round(routeData.route_summary.performance.average_departure_delay)}s
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Summary Metrics */}
        {routeData.route_summary && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {routeData.route_summary.stop_topology && (
              <>
                <MetricCard 
                  title="Parent Stations" 
                  value={routeData.route_summary.stop_topology.total_parent_stations}
                  subtitle="stations served"
                  bgColor="bg-blue-50"
                  iconColor="text-blue-600"
                />
                <MetricCard 
                  title="Stop IDs" 
                  value={routeData.route_summary.stop_topology.total_stop_ids}
                  subtitle="physical stops"
                  bgColor="bg-green-50"
                  iconColor="text-green-600"
                />
              </>
            )}
            {routeData.route_summary.direction_topology && (
              <>
                <MetricCard 
                  title="Directions" 
                  value={routeData.route_summary.direction_topology.total_directions}
                  subtitle="route directions"
                  bgColor="bg-purple-50"
                  iconColor="text-purple-600"
                />
                <MetricCard 
                  title="Trip Instances" 
                  value={routeData.route_summary.direction_topology.total_trip_instances}
                  subtitle="total trips"
                  bgColor="bg-orange-50"
                  iconColor="text-orange-600"
                />
              </>
            )}
          </div>
        )}

        {/* Topology Breakdown */}
        {routeData.route_summary && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {routeData.route_summary.stop_topology && (
              <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Stop Topology</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Total Labels</span>
                    <span className="text-sm font-medium text-gray-900">
                      {routeData.route_summary.stop_topology.total_labels}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Total Violations</span>
                    <span className={`text-sm font-medium ${
                      routeData.route_summary.stop_topology.total_violations > 0 
                        ? 'text-red-600' 
                        : 'text-green-600'
                    }`}>
                      {routeData.route_summary.stop_topology.total_violations}
                    </span>
                  </div>
                  {routeData.route_summary.stop_topology.labels_counts_by_type && (
                    <>
                      <div className="flex justify-between items-center pl-4">
                        <span className="text-xs text-gray-500">Parent Station Labels</span>
                        <span className="text-xs font-medium text-gray-700">
                          {routeData.route_summary.stop_topology.labels_counts_by_type.parent_station}
                        </span>
                      </div>
                      <div className="flex justify-between items-center pl-4">
                        <span className="text-xs text-gray-500">Stop ID Labels</span>
                        <span className="text-xs font-medium text-gray-700">
                          {routeData.route_summary.stop_topology.labels_counts_by_type.stop_id}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {routeData.route_summary.direction_topology && (
              <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Direction Topology</h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Total Labels</span>
                    <span className="text-sm font-medium text-gray-900">
                      {routeData.route_summary.direction_topology.total_labels}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Total Violations</span>
                    <span className={`text-sm font-medium ${
                      routeData.route_summary.direction_topology.total_violations > 0 
                        ? 'text-red-600' 
                        : 'text-green-600'
                    }`}>
                      {routeData.route_summary.direction_topology.total_violations}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Canonical Share</span>
                    <span className="text-sm font-medium text-gray-900">
                      {(routeData.route_summary.direction_topology.canonical_share * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Performance Metrics */}
        {routeData.route_summary?.performance && (
          <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {routeData.route_summary.performance.overall_on_time_rate}%
                </div>
                <div className="text-sm text-gray-500">On Time</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-600">
                  {routeData.route_summary.performance.overall_too_early_rate}%
                </div>
                <div className="text-sm text-gray-500">Too Early</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {routeData.route_summary.performance.overall_too_late_rate}%
                </div>
                <div className="text-sm text-gray-500">Too Late</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round(routeData.route_summary.performance.average_departure_delay)}s
                </div>
                <div className="text-sm text-gray-500">Avg Delay</div>
              </div>
            </div>
          </div>
        )}

        {/* Regulatory Stops */}
        {routeData.route_summary?.regulatory_stops && (
          <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Regulatory Information</h3>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800 border border-amber-200">
                  {routeData.route_summary.regulatory_stops.total_regulatory_stops} Regulatory Stops
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Directions Detail Section */}
        {routeData.directions && Object.keys(routeData.directions).length > 0 && (
          <div className="bg-white shadow-sm rounded-lg border border-gray-200">
            {/* Header with Show Directions Button */}
            <div className="px-6 py-4 flex justify-between items-center border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Direction Breakdown</h2>
              <button
                onClick={toggleDirectionsVisibility}
                className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
              >
                {showDirections ? "Hide Directions" : "Show Directions"}
              </button>
            </div>

            {/* Conditionally render directions */}
            {showDirections && (
              <div className="p-6 space-y-6">
                {Object.entries(routeData.directions).map(([directionId, direction]: [string, any]) => (
                  <DirectionDetailCard 
                    key={directionId} 
                    direction={direction} 
                    directionId={directionId}
                    routeId={routeId}
                    globalData={globalData}
                    isExpanded={expandedDirections.has(directionId)}
                    onToggleExpansion={() => toggleDirectionExpansion(directionId)}
                    showPunctualityFlow={showPunctualityFlow[directionId] || false}
                    onTogglePunctualityFlow={() => togglePunctualityFlow(directionId)}
                    selectedTimeType={selectedTimeType[directionId] || 'day'}
                    selectedMetric={selectedMetric[directionId] || 'on_time'}
                    onUpdateTimeType={(timeType: string) => updateTimeType(directionId, timeType)}
                    onUpdateMetric={(metric: string) => updateMetric(directionId, metric)}
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

export default RouteLayout