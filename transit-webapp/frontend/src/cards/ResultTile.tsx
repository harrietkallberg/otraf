import React, { useState } from 'react'
import CompactHistogram from './CompactHistogram'

interface ResultCombination {
  routeId: string
  directionId: string
  stopId: string
  timeType: string
  stopName: string
  routeName: string
  performanceKey: string
  hasPerformance: boolean
  labelCount: number
  violationCount: number
  performanceData?: any
  labels: any[]
  violations: any[]
}

interface ResultTileProps {
  result: ResultCombination
  index: number
  globalData: any
}

const ResultTile: React.FC<ResultTileProps> = ({ result, index, globalData }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  
  const toggleExpansion = () => {
    setIsExpanded(!isExpanded)
  }

  const performanceData = result.performanceData

  return (
    <div className="border p-4 rounded shadow-sm bg-white">
      {/* Card Header */}
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-gray-500">
          {result.timeType.replace('_', ' ').toUpperCase()}
        </span>
        
        {/* Summary Badges */}
        <div className="flex items-center space-x-2">
          {performanceData?.analytics?.punctuality && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              {performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}% on time
            </span>
          )}
          {performanceData?.analytics?.is_regulatory_stop && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
              Regulatory
            </span>
          )}
          {result.labelCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              {result.labelCount} labels
            </span>
          )}
          {result.violationCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
              {result.violationCount} violations
            </span>
          )}
        </div>
      </div>
      
      <div className="text-lg font-semibold mb-1">
        Route {result.routeName} - {result.stopName}
      </div>
      
      <div className="text-sm text-gray-600 mb-3">
        Direction {result.directionId} • Stop ID: {result.stopId}
      </div>

      {/* Performance Summary */}
      {performanceData?.analytics?.punctuality && (
        <div className="text-sm text-gray-600 space-x-4 mb-3">
          <span>
            Sample: {performanceData.analytics.punctuality.sample_size.toLocaleString()}
          </span>
          <span>
            Mean delay: {performanceData.analytics.punctuality.basic_statistics.mean_delay.toFixed(1)}s
          </span>
        </div>
      )}

      {/* Show Details / Hide Details Button */}
      <button
        onClick={toggleExpansion}
        className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
      >
        {isExpanded ? 'Hide Details' : 'Show Details'}
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <ExpandedResultDetails 
            result={result}
            globalData={globalData}
            performanceData={performanceData}
          />
        </div>
      )}
    </div>
  )
}

// Expanded Result Details Component
const ExpandedResultDetails: React.FC<{
  result: ResultCombination
  globalData: any
  performanceData: any
}> = ({ result, globalData, performanceData }) => {
  
  return (
    <div className="space-y-6">
      
      {/* Performance Details */}
      {performanceData?.analytics && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Performance Analysis</h4>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Punctuality Breakdown */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h5 className="text-xs font-medium text-gray-600 mb-3">PUNCTUALITY DISTRIBUTION</h5>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-green-600">On Time</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-yellow-600">Too Early</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.too_early}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-red-600">Too Late</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.too_late}%</span>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h5 className="text-xs font-medium text-gray-600 mb-3">DELAY STATISTICS</h5>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Mean</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.basic_statistics.mean_delay.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Median</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.basic_statistics.median_delay}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Sample Size</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.sample_size.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Compact Histogram */}
            <CompactHistogram performanceData={performanceData} />
          </div>
        </div>
      )}

      {/* Labels */}
      {result.labels.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Labels ({result.labels.length})</h4>
          <div className="space-y-2">
            {result.labels.map((label, index) => (
              <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {label.label_type}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{label.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Violations */}
      {result.violations.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Violations ({result.violations.length})</h4>
          <div className="space-y-2">
            {result.violations.map((violation, index) => {
              const getSeverityColor = (severity: number) => {
                if (severity >= 5) return 'bg-red-100 text-red-800 border-red-200'
                if (severity >= 3) return 'bg-orange-100 text-orange-800 border-orange-200'
                return 'bg-yellow-100 text-yellow-800 border-yellow-200'
              }

              return (
                <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-3">
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
                      <p className="text-sm text-gray-700">{violation.description}</p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* No Additional Data */}
      {result.labels.length === 0 && result.violations.length === 0 && (
        <div className="text-center py-6 text-gray-500 text-sm">
          No labels or violations found for this combination
        </div>
      )}
    </div>
  )
}

export default ResultTile