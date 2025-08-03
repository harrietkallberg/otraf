import React from 'react'
import PunctualityFlowChart from './PunctualityChart'

interface DirectionDetailCardProps {
  direction: any
  directionId: string
  routeId: string
  globalData: any
  isExpanded: boolean
  onToggleExpansion: () => void
  showPunctualityFlow: boolean
  onTogglePunctualityFlow: () => void
  selectedTimeType: string
  selectedMetric: string
  onUpdateTimeType: (timeType: string) => void
  onUpdateMetric: (metric: string) => void
}

const DirectionDetailCard: React.FC<DirectionDetailCardProps> = ({ 
  direction, 
  directionId, 
  routeId, 
  globalData,
  isExpanded, 
  onToggleExpansion,
  showPunctualityFlow,
  onTogglePunctualityFlow,
  selectedTimeType,
  selectedMetric,
  onUpdateTimeType,
  onUpdateMetric
}) => {
  
  // Get stops from stops_in_direction
  const stops = direction.stops_in_direction || {}
  const stopCount = Object.keys(stops).length
  
  // Calculate totals from direction_summary
  const totalLabels = (direction.direction_summary?.stop_topology?.labels_by_type?.parent_station || 0) + 
                     (direction.direction_summary?.stop_topology?.labels_by_type?.stop_id || 0) +
                     (direction.direction_summary?.direction_topology?.labels_by_type?.direction_id || 0) +
                     (direction.direction_summary?.direction_topology?.labels_by_type?.stop_id || 0)
                     
  const totalViolations = (direction.direction_summary?.stop_topology?.violations_by_type?.parent_station || 0) + 
                         (direction.direction_summary?.stop_topology?.violations_by_type?.stop_id || 0) +
                         (direction.direction_summary?.direction_topology?.violations_by_type?.direction_id || 0) +
                         (direction.direction_summary?.direction_topology?.violations_by_type?.stop_id || 0)

  const totalRegulatoryStops = direction.direction_summary?.regulatory_stops?.regulatory_stop_ids || 0
  const totalPerformanceAnalytics = direction.direction_summary?.performance?.available_performace_analytics || 0

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <div className="bg-gray-50 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-sm font-bold text-blue-700">{directionId}</span>
              </div>
            </div>
            <div className="flex-1">
              <h3 className="text-base font-semibold text-gray-900">Direction {directionId}</h3>
              <p className="text-sm text-gray-500">{stopCount} stops in sequence</p>
              <div className="mt-2 flex items-center space-x-4 text-xs text-gray-600">
                <span>{totalPerformanceAnalytics} analytics available</span>
                {totalRegulatoryStops > 0 && (
                  <span className="text-amber-600">{totalRegulatoryStops} regulatory stops</span>
                )}
              </div>
              
              <div className="mt-2 flex items-center space-x-2">
                <button
                  onClick={onToggleExpansion}
                  className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
                >
                  {isExpanded ? (
                    <>
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7" />
                      </svg>
                      Hide Stops
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                      </svg>
                      Show Stops
                    </>
                  )}
                </button>
                
                {totalPerformanceAnalytics > 0 && (
                  <button
                    onClick={onTogglePunctualityFlow}
                    className="inline-flex items-center text-sm font-medium text-green-600 hover:text-green-700 focus:outline-none transition-colors"
                  >
                    {showPunctualityFlow ? 'Hide Chart' : 'Show Chart'}
                  </button>
                )}
              </div>
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

      {/* Punctuality Flow Chart */}
      {showPunctualityFlow && totalPerformanceAnalytics > 0 && (
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-medium text-gray-700">Performance Flow Chart</h4>
            <div className="flex items-center space-x-4">
              {/* Time Type Selector */}
              <select
                value={selectedTimeType}
                onChange={(e) => onUpdateTimeType(e.target.value)}
                className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="am_rush">AM Rush</option>
                <option value="day">Day</option>
                <option value="pm_rush">PM Rush</option>
                <option value="night">Night</option>
                <option value="weekend">Weekend</option>
              </select>
              
              {/* Metric Selector */}
              <select
                value={selectedMetric}
                onChange={(e) => onUpdateMetric(e.target.value)}
                className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="on_time">On Time</option>
                <option value="too_late">Too Late</option>
                <option value="too_early">Too Early</option>
              </select>
            </div>
          </div>
          
          <PunctualityFlowChart
            routeId={routeId}
            directionId={directionId}
            direction={direction}
            globalData={globalData}
            selectedTimeType={selectedTimeType}
            selectedMetric={selectedMetric}
          />
        </div>
      )}

      {isExpanded && (
        <div className="px-6 py-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-4">Stop Sequence</h4>
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
      )}
    </div>
  )
}

export default DirectionDetailCard