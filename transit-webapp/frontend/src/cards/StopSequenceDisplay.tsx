import React from 'react'

interface StopSequenceDisplayProps {
  direction: any
}

const StopSequenceDisplay: React.FC<StopSequenceDisplayProps> = ({ direction }) => {
  
  if (!direction.canonical_patterns) {
    return <div className="text-gray-500 text-sm">No stop sequence data available</div>
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="space-y-3">
        {Object.entries(direction.canonical_patterns)
          .sort(([a], [b]) => parseInt(a) - parseInt(b))
          .map(([position, stop]: [string, any]) => (
          <StopSequenceItem 
            key={position} 
            position={position} 
            stop={stop}
          />
        ))}
      </div>
    </div>
  )
}

// Stop Sequence Item Component
const StopSequenceItem: React.FC<{ 
  position: string
  stop: any
}> = ({ position, stop }) => (
  <div className="bg-gray-50 rounded-lg p-4">
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            <div className="w-8 h-8 bg-white rounded-full border-2 border-blue-200 flex items-center justify-center">
              <span className="text-xs font-medium text-blue-600">{position}</span>
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
      </div>
    </div>
  </div>
)

export default StopSequenceDisplay