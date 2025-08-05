import React, { useState, useContext, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { StopDetailsView, Badge } from '../components/shared'
import { ProgressiveSearchFilters, FilterConfig } from '../components/shared/ProgressiveSearchFilters'
import { useProgressiveSearch, FilterValidationRule } from '../hooks/useProgressiveSearch'

interface ExplorePageProps {}

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

const ExplorePage: React.FC<ExplorePageProps> = () => {
  const { routeId: urlRouteId, directionId: urlDirectionId, stopId: urlStopId, timeType: urlTimeType } = useParams()
  const globalData = useContext(GlobalDataContext)
  const [showHelp, setShowHelp] = useState(false)
  
  // All combinations from performance data
  const [allCombinations, setAllCombinations] = useState<ResultCombination[]>([])

  // Extract all combinations when globalData is available
  useEffect(() => {
    if (!globalData) return

    // Convert labels and violations arrays to objects keyed by entity_key if needed
    const labelsObj = Array.isArray(globalData.labels) 
      ? Object.fromEntries(globalData.labels.map((label: any) => [label.entity_key, label]))
      : globalData.labels || {}
      
    const violationsObj = Array.isArray(globalData.violations)
      ? Object.fromEntries(globalData.violations.map((violation: any) => [violation.entity_key, violation]))
      : globalData.violations || {}

    const combinations: ResultCombination[] = []

    // Extract all combinations from performance data
    Object.keys(globalData.performance || {}).forEach(key => {
      const match = key.match(/^performance_([^_]+(?:_[^_]+)*)_direction_id_stop_id_time_type_(\d+)_([^_]+)_(.+)$/)
      if (match) {
        const [, routeId, directionId, stopId, timeType] = match
        const performanceData = globalData.performance[key]
        
        // Find parent station by looking up the stop_id in global stops data
        let parentStation = performanceData?.parent_station
        let allStopIdsForThisStation = [stopId]
        
        if (!parentStation && globalData.stops) {
          for (const [parentStationId, stationData] of Object.entries(globalData.stops)) {
            if (stationData.stop_ids && stationData.stop_ids.includes(stopId)) {
              parentStation = parentStationId
              allStopIdsForThisStation = stationData.stop_ids
              break
            }
          }
        }
        
        // Build possible label and violation keys
        const possibleLabelKeys = [
          `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${stopId}`,
          `direction_topology_${routeId}_direction_id_${directionId}`,
          `regulatory_stops_${routeId}_direction_id_stop_id_${directionId}_${stopId}`,
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_${parentStation}`] : []),
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${stopId}`] : []),
          ...(parentStation ? allStopIdsForThisStation
            .filter(sid => sid !== stopId)
            .map(sid => `stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${sid}`) : []),
          ...allStopIdsForThisStation
            .filter(sid => sid !== stopId)
            .map(sid => `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${sid}`)
        ]

        const possibleViolationKeys = [
          `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${stopId}`,
          `direction_topology_${routeId}_direction_id_${directionId}`,
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_${parentStation}`] : []),
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${stopId}`] : []),
          ...(parentStation ? allStopIdsForThisStation
            .filter(sid => sid !== stopId)
            .map(sid => `stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${sid}`) : []),
          ...allStopIdsForThisStation
            .filter(sid => sid !== stopId)
            .map(sid => `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${sid}`)
        ]
        
        // Get actual label and violation data
        const labels = possibleLabelKeys
          .filter(labelKey => labelsObj[labelKey])
          .map(labelKey => ({ key: labelKey, ...labelsObj[labelKey] }))

        const violations = possibleViolationKeys
          .filter(violationKey => violationsObj[violationKey])
          .map(violationKey => ({ key: violationKey, ...violationsObj[violationKey] }))

        combinations.push({
          routeId,
          directionId,
          stopId,
          timeType,
          stopName: performanceData?.stop_name || stopId,
          routeName: String(globalData.routes?.[routeId]?.route_short_name || routeId),
          performanceKey: key,
          hasPerformance: !!performanceData,
          labelCount: labels.length,
          violationCount: violations.length,
          performanceData,
          labels,
          violations
        })
      }
    })

    setAllCombinations(combinations)
  }, [globalData])

  // Define filter rules for progressive search
  const filterRules: FilterValidationRule<ResultCombination>[] = [
    {
      key: 'routeId',
      validateItem: (item, filters) => !filters.routeId || item.routeId === filters.routeId,
      extractOptions: (item) => item.routeId,
    },
    {
      key: 'directionId',
      validateItem: (item, filters) => !filters.directionId || item.directionId === filters.directionId,
      extractOptions: (item) => item.directionId,
      dependencies: ['stopId', 'timeType']
    },
    {
      key: 'stopId',
      validateItem: (item, filters) => !filters.stopId || item.stopId === filters.stopId,
      extractOptions: (item) => item.stopId,
      dependencies: ['timeType']
    },
    {
      key: 'timeType',
      validateItem: (item, filters) => !filters.timeType || item.timeType === filters.timeType,
      extractOptions: (item) => item.timeType,
    }
  ];

  // Use the progressive search hook
  const {
    filters,
    setFilter,
    filteredData: filteredResults,
    availableOptions
  } = useProgressiveSearch({
    data: allCombinations,
    initialFilters: {
      routeId: urlRouteId || '',
      directionId: urlDirectionId || '',
      stopId: urlStopId || '',
      timeType: urlTimeType || ''
    },
    filterRules,
    hasValidData: (item) => item.hasPerformance
  });

  // Create filter configurations
  const filterConfigs: FilterConfig[] = [
    {
      key: 'routeId',
      label: 'Route',
      placeholder: '(all routes)',
      value: filters.routeId || '',
      onChange: (value) => setFilter('routeId', value),
      options: Array.from(availableOptions.routeId || []).sort().map(routeId => ({
        value: routeId,
        label: `Route ${globalData?.routes?.[routeId]?.route_short_name || routeId}`
      }))
    },
    {
      key: 'directionId',
      label: 'Direction',
      placeholder: '(both directions)',
      value: filters.directionId || '',
      onChange: (value) => setFilter('directionId', value),
      options: [
        { value: '0', label: 'Direction 0' },
        { value: '1', label: 'Direction 1' }
      ].filter(option => availableOptions.directionId?.has(option.value))
    },
    {
      key: 'stopId',
      label: 'Stop',
      placeholder: '(all stops)',
      value: filters.stopId || '',
      onChange: (value) => setFilter('stopId', value),
      options: Array.from(availableOptions.stopId || []).sort().map(stopId => {
        const stopName = allCombinations.find(c => c.stopId === stopId)?.stopName || stopId;
        return {
          value: stopId,
          label: `${stopName} (${stopId})`
        };
      })
    },
    {
      key: 'timeType',
      label: 'Time Type',
      placeholder: '(all time types)',
      value: filters.timeType || '',
      onChange: (value) => setFilter('timeType', value),
      options: (globalData?.time_types || [])
        .filter((timeType: string) => availableOptions.timeType?.has(timeType))
        .map((timeType: string) => ({
          value: timeType,
          label: timeType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
        }))
    }
  ];

  // Sort filtered results
  const sortedResults = filteredResults.sort((a, b) => {
    const aRouteName = String(a.routeName || a.routeId)
    const bRouteName = String(b.routeName || b.routeId)
    const aStopName = String(a.stopName || a.stopId)
    const bStopName = String(b.stopName || b.stopId)
    const aDirectionId = String(a.directionId)
    const bDirectionId = String(b.directionId)
    const aTimeType = String(a.timeType)
    const bTimeType = String(b.timeType)
    
    return aRouteName.localeCompare(bRouteName) || 
           aDirectionId.localeCompare(bDirectionId) ||
           aStopName.localeCompare(bStopName) ||
           aTimeType.localeCompare(bTimeType)
  });

  if (!globalData) {
    return <div className="p-6">Loading explore data...</div>
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header Section with Help */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <h1 className="text-3xl font-bold">Explore Logs</h1>
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="w-7 h-7 bg-blue-100 hover:bg-blue-200 rounded-full flex items-center justify-center transition-colors duration-200"
            title="Help"
          >
            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>
        <div className="text-sm text-gray-500">
          {allCombinations.length} combinations found
        </div>
      </div>

      {/* Help Section */}
      {showHelp && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <svg className="w-5 h-5 text-blue-600 mt-0.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">Explore Logs Help</h3>
              <p className="text-sm text-blue-800 leading-relaxed">
                This page provides a comprehensive view of all performance data, labels, and violations across routes, directions, stops, and time periods. Each combination represents a unique route-direction-stop-time scenario with associated performance metrics, regulatory labels, and topology violations. Use the filters to narrow down to specific combinations and click "Show Details" to see detailed performance analytics, labels, and violation information for each scenario.
              </p>
              <button
                onClick={() => setShowHelp(false)}
                className="mt-4 text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                Got it, hide help
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Progressive Search Filters */}
      <ProgressiveSearchFilters
        title="Filter Results"
        subtitle={`Showing ${sortedResults.length} of ${allCombinations.length} combinations`}
        filters={filterConfigs}
        className="mb-6"
      />

      {/* Results */}
      <div className="space-y-4">
        {sortedResults.map((result, idx) => (
          <ExploreResultCard 
            key={idx}
            result={result}
            globalData={globalData}
          />
        ))}

        {sortedResults.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500 text-lg">No matching combinations found.</p>
            <p className="text-gray-400 text-sm mt-2">
              Try adjusting your filters to see available data.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

// Refactored component using shared components
const ExploreResultCard: React.FC<{
  result: ResultCombination;
  globalData: any;
}> = ({ result, globalData }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Check if this is a regulatory stop
  const isRegulatory = result.performanceData?.analytics?.is_regulatory_stop || 
                       result.labels.some(label => label.key?.includes('regulatory_stops'));

  return (
    <div className={`border rounded-lg transition-all ${
      isRegulatory 
        ? 'border-orange-200 bg-orange-50' 
        : 'border-gray-200 bg-white'
    }`}>
      <div className="p-4">
        {/* Card Header */}
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-500">
            {result.timeType.replace('_', ' ').toUpperCase()}
          </span>
          
          {/* Summary Badges using shared Badge component */}
          <div className="flex items-center space-x-2">
            {result.performanceData?.analytics?.punctuality && (
              <Badge 
                count={0} 
                type="analytics" 
                customText={`${result.performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}% on time`}
                size="sm"
              />
            )}
            {isRegulatory && (
              <Badge count={0} type="regulatory" customText="Regulatory" size="sm" />
            )}
            {result.labelCount > 0 && (
              <Badge count={result.labelCount} type="labels" size="sm" />
            )}
            {result.violationCount > 0 && (
              <Badge count={result.violationCount} type="violations" size="sm" />
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
        {result.performanceData?.analytics?.punctuality && (
          <div className="text-sm text-gray-600 space-x-4 mb-3">
            <span>
              Sample: {result.performanceData.analytics.punctuality.sample_size.toLocaleString()}
            </span>
            <span>
              Mean delay: {result.performanceData.analytics.punctuality.basic_statistics.mean_delay.toFixed(1)}s
            </span>
          </div>
        )}

        {/* Show Details / Hide Details Button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
        >
          {isExpanded ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {/* Expanded Details - Using shared StopDetailsView component */}
      {isExpanded && (
        <div className="border-t border-gray-200 p-4 bg-white">
          <StopDetailsView 
            labels={result.labels}
            violations={result.violations}
            performanceData={result.performanceData}
            size="lg"
          />
        </div>
      )}
    </div>
  );
};

export default ExplorePage