import React from 'react'

interface SummaryCardProps {
  title: string
  data: {
    stop_topology?: {
      labels_by_type: { parent_station: number; stop_id: number }
      violations_by_type: { parent_station: number; stop_id: number }
    }
    direction_topology?: {
      labels_by_type: { direction_id: number; stop_id: number }
      violations_by_type: { direction_id: number; stop_id: number }
    }
    regulatory_stops?: {
      regulatory_stop_ids: number
    }
    performance?: {
      available_performace_analytics: number
    }
  }
}

const SummaryCard: React.FC<SummaryCardProps> = ({ title, data }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Stop Topology */}
      {data.stop_topology && (
        <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Stop Topology</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Parent Station Labels</span>
              <span className="text-sm font-medium text-gray-900">
                {data.stop_topology.labels_by_type.parent_station}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Stop ID Labels</span>
              <span className="text-sm font-medium text-gray-900">
                {data.stop_topology.labels_by_type.stop_id}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Parent Station Violations</span>
              <span className={`text-sm font-medium ${
                data.stop_topology.violations_by_type.parent_station > 0 
                  ? 'text-red-600' 
                  : 'text-green-600'
              }`}>
                {data.stop_topology.violations_by_type.parent_station}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Stop ID Violations</span>
              <span className={`text-sm font-medium ${
                data.stop_topology.violations_by_type.stop_id > 0 
                  ? 'text-red-600' 
                  : 'text-green-600'
              }`}>
                {data.stop_topology.violations_by_type.stop_id}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Direction Topology */}
      {data.direction_topology && (
        <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Direction Topology</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Direction ID Labels</span>
              <span className="text-sm font-medium text-gray-900">
                {data.direction_topology.labels_by_type.direction_id}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Stop ID Labels</span>
              <span className="text-sm font-medium text-gray-900">
                {data.direction_topology.labels_by_type.stop_id}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Direction ID Violations</span>
              <span className={`text-sm font-medium ${
                data.direction_topology.violations_by_type.direction_id > 0 
                  ? 'text-red-600' 
                  : 'text-green-600'
              }`}>
                {data.direction_topology.violations_by_type.direction_id}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Stop ID Violations</span>
              <span className={`text-sm font-medium ${
                data.direction_topology.violations_by_type.stop_id > 0 
                  ? 'text-red-600' 
                  : 'text-green-600'
              }`}>
                {data.direction_topology.violations_by_type.stop_id}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Additional Info */}
      {(data.regulatory_stops || data.performance) && (
        <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Additional Information</h3>
          <div className="flex items-center space-x-6">
            {data.regulatory_stops && (
              <div className="flex items-center space-x-2">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800 border border-amber-200">
                  {data.regulatory_stops.regulatory_stop_ids} Regulatory Stops
                </span>
              </div>
            )}
            {data.performance && (
              <div className="flex items-center space-x-2">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200">
                  {data.performance.available_performace_analytics} Performance Analytics
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default SummaryCard