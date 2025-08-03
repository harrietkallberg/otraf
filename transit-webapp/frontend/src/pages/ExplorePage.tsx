import React, { useState, useContext, useMemo, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import ResultTile from '../cards/ResultTile'

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
  
  // State for search filters
  const [selectedRouteId, setSelectedRouteId] = useState(urlRouteId || '')
  const [selectedDirectionId, setSelectedDirectionId] = useState(urlDirectionId || '')
  const [selectedStopId, setSelectedStopId] = useState(urlStopId || '')
  const [selectedTimeType, setSelectedTimeType] = useState(urlTimeType || '')

  // All combinations from performance data
  const [allCombinations, setAllCombinations] = useState<ResultCombination[]>([])

  // Extract all combinations when globalData is available
  useEffect(() => {
    if (!globalData) return

    // Debug: Let's see what's actually in the data
    console.log('GlobalData structure:', {
      hasLabels: !!globalData.labels,
      hasViolations: !!globalData.violations,
      hasPerformance: !!globalData.performance,
      labelsType: Array.isArray(globalData.labels) ? 'array' : 'object',
      violationsType: Array.isArray(globalData.violations) ? 'array' : 'object',
      labelKeys: globalData.labels ? Object.keys(globalData.labels).slice(0, 5) : [],
      violationKeys: globalData.violations ? Object.keys(globalData.violations).slice(0, 5) : [],
      performanceKeys: globalData.performance ? Object.keys(globalData.performance).slice(0, 5) : [],
      sampleLabel: globalData.labels ? globalData.labels[Object.keys(globalData.labels)[0]] : null,
      sampleViolation: globalData.violations ? globalData.violations[Object.keys(globalData.violations)[0]] : null
    })

    // Convert labels and violations arrays to objects keyed by entity_key if needed
    const labelsObj = Array.isArray(globalData.labels) 
      ? Object.fromEntries(globalData.labels.map((label: any) => [label.entity_key, label]))
      : globalData.labels || {}
      
    const violationsObj = Array.isArray(globalData.violations)
      ? Object.fromEntries(globalData.violations.map((violation: any) => [violation.entity_key, violation]))
      : globalData.violations || {}

    console.log('Converted data:', {
      labelsObjKeys: Object.keys(labelsObj).slice(0, 5),
      violationsObjKeys: Object.keys(violationsObj).slice(0, 5)
    })

    const combinations: ResultCombination[] = []

    // Extract all combinations from performance data
    Object.keys(globalData.performance || {}).forEach(key => {
      // Better regex that specifically looks for the pattern
      const match = key.match(/^performance_([^_]+(?:_[^_]+)*)_direction_id_stop_id_time_type_(\d+)_([^_]+)_(.+)$/)
      if (match) {
        const [, routeId, directionId, stopId, timeType] = match
        
        // Debug the regex parsing for first few keys
        if (combinations.length < 3) {
          console.log(`Parsing key: ${key}`)
          console.log(`Parsed:`, { routeId, directionId, stopId, timeType })
        }
        
        // Get performance data
        const performanceData = globalData.performance[key]
        
        // Find parent station by looking up the stop_id in global stops data
        let parentStation = performanceData?.parent_station
        let allStopIdsForThisStation = [stopId] // Start with current stop_id
        
        if (!parentStation && globalData.stops) {
          // Search through all parent stations to find which one contains this stop_id
          for (const [parentStationId, stationData] of Object.entries(globalData.stops)) {
            if (stationData.stop_ids && stationData.stop_ids.includes(stopId)) {
              parentStation = parentStationId
              allStopIdsForThisStation = stationData.stop_ids // Get ALL stop_ids for this parent station
              break
            }
          }
        }
        
        // Build standardized keys following the logger's build_entity_key pattern:
        // {domain}_{route_id}_{kind}_{identifier}
        // Include ALL hierarchical levels that could apply to this combination
        const possibleLabelKeys = [
          // Direction + Stop level for current stop (most specific)
          `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${stopId}`,
          // Direction level (applies to whole direction)
          `direction_topology_${routeId}_direction_id_${directionId}`,
          // Regulatory stops for current stop
          `regulatory_stops_${routeId}_direction_id_stop_id_${directionId}_${stopId}`,
          // Parent station level (if we have parent station - applies to whole parent station)
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_${parentStation}`] : []),
          // Parent station + stop level for current stop (if we have parent station)
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${stopId}`] : []),
          // Parent station + stop level for ALL other stop_ids in this parent station
          ...(parentStation ? allStopIdsForThisStation
            .filter(sid => sid !== stopId) // Exclude current stop_id as we already have it
            .map(sid => `stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${sid}`) : []),
          // Direction + Stop level for ALL other stop_ids in this parent station
          ...allStopIdsForThisStation
            .filter(sid => sid !== stopId) // Exclude current stop_id as we already have it
            .map(sid => `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${sid}`)
        ]

        const possibleViolationKeys = [
          // Direction + Stop level for current stop (most specific)
          `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${stopId}`,
          // Direction level (applies to whole direction)
          `direction_topology_${routeId}_direction_id_${directionId}`,
          // Parent station level (if we have parent station - applies to whole parent station)
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_${parentStation}`] : []),
          // Parent station + stop level for current stop (if we have parent station)
          ...(parentStation ? [`stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${stopId}`] : []),
          // Parent station + stop level for ALL other stop_ids in this parent station
          ...(parentStation ? allStopIdsForThisStation
            .filter(sid => sid !== stopId) // Exclude current stop_id as we already have it
            .map(sid => `stop_topology_${routeId}_parent_station_stop_id_${parentStation}_${sid}`) : []),
          // Direction + Stop level for ALL other stop_ids in this parent station
          ...allStopIdsForThisStation
            .filter(sid => sid !== stopId) // Exclude current stop_id as we already have it
            .map(sid => `direction_topology_${routeId}_direction_id_stop_id_${directionId}_${sid}`)
        ]
        
        // Debug for first few combinations
        if (combinations.length < 3) {
          console.log(`Debug combination ${combinations.length + 1}:`, {
            routeId, directionId, stopId, timeType, parentStation,
            possibleLabelKeys,
            possibleViolationKeys,
            foundLabels: possibleLabelKeys.filter(key => labelsObj[key]),
            foundViolations: possibleViolationKeys.filter(key => violationsObj[key])
          })
        }
        
        // Get actual label and violation data using the standardized keys
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

    console.log('Setting explore combinations:', combinations.length, 'total')
    console.log('Sample combinations with labels/violations:', 
      combinations.filter(c => c.labelCount > 0 || c.violationCount > 0).slice(0, 3)
    )
    setAllCombinations(combinations)
  }, [globalData])

  const hasLetters = (s: string) => /\D/.test(s)

  // Smart filter clearing - only clear if current selection becomes invalid
  useEffect(() => {
    if (!selectedRouteId && !selectedDirectionId && !selectedStopId && !selectedTimeType) return
    
    const validRoutes = new Set<string>()
    const validDirections = new Set<string>()
    const validStops = new Set<string>()
    const validTimeTypes = new Set<string>()
    
    allCombinations.forEach(combination => {
      if (!combination.hasPerformance) return

      const routeMatch = !selectedRouteId || combination.routeId === selectedRouteId
      const directionMatch = !selectedDirectionId || combination.directionId === selectedDirectionId
      const stopMatch = !selectedStopId || combination.stopId === selectedStopId
      const timeMatch = !selectedTimeType || combination.timeType === selectedTimeType

      if (directionMatch && stopMatch && timeMatch) validRoutes.add(combination.routeId)
      if (routeMatch && stopMatch && timeMatch) validDirections.add(combination.directionId)
      if (routeMatch && directionMatch && timeMatch) validStops.add(combination.stopId)
      if (routeMatch && directionMatch && stopMatch) validTimeTypes.add(combination.timeType)
    })

    // Only clear if current selection is no longer valid
    if (selectedRouteId && !validRoutes.has(selectedRouteId)) {
      setSelectedRouteId('')
      setSelectedDirectionId('')
      setSelectedStopId('')
      setSelectedTimeType('')
    } else if (selectedDirectionId && !validDirections.has(selectedDirectionId)) {
      setSelectedDirectionId('')
      setSelectedStopId('')
      setSelectedTimeType('')
    } else if (selectedStopId && !validStops.has(selectedStopId)) {
      setSelectedStopId('')
      setSelectedTimeType('')
    } else if (selectedTimeType && !validTimeTypes.has(selectedTimeType)) {
      setSelectedTimeType('')
    }
  }, [selectedRouteId, selectedDirectionId, selectedStopId, selectedTimeType, allCombinations])

  // Progressive filtering and results - single pass approach
  const { availableOptions, filteredResults } = useMemo(() => {
    const routes = new Set<string>()
    const directions = new Set<string>()
    const stops = new Set<string>()
    const timeTypes = new Set<string>()
    const filtered: ResultCombination[] = []

    allCombinations.forEach(combination => {
      if (!combination.hasPerformance) return

      // Apply current filters
      const routeMatch = !selectedRouteId || combination.routeId === selectedRouteId
      const directionMatch = !selectedDirectionId || combination.directionId === selectedDirectionId
      const stopMatch = !selectedStopId || combination.stopId === selectedStopId
      const timeMatch = !selectedTimeType || combination.timeType === selectedTimeType

      // If matches all current filters, include in results
      if (routeMatch && directionMatch && stopMatch && timeMatch) {
        filtered.push(combination)
      }

      // Collect available options based on partial matches (progressive filtering)
      if (directionMatch && stopMatch && timeMatch) routes.add(combination.routeId)
      if (routeMatch && stopMatch && timeMatch) directions.add(combination.directionId)
      if (routeMatch && directionMatch && timeMatch) stops.add(combination.stopId)
      if (routeMatch && directionMatch && stopMatch) timeTypes.add(combination.timeType)
    })

    return {
      availableOptions: {
        routes: Array.from(routes).sort(),
        directions: Array.from(directions).sort(),
        stops: Array.from(stops).sort(),
        timeTypes: Array.from(timeTypes).sort()
      },
      filteredResults: filtered.sort((a, b) => {
        // Ensure all values are strings before comparing
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
      })
    }
  }, [allCombinations, selectedRouteId, selectedDirectionId, selectedStopId, selectedTimeType])

  if (!globalData) {
    return <div className="p-6">Loading explore data...</div>
  }

  return (
    <div className="p-6">
      
      <h1 className="text-2xl font-bold mb-4"> Explore Logs of Performance, Delays, Labels and Violations</h1>
      
      <div className="mb-4 text-sm text-gray-600">
        Found {allCombinations.length} combinations total, {filteredResults.length} matching filters
      </div>

      {/* Progressive Search Filters */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block mb-1">Route</label>
          <select
            value={selectedRouteId}
            onChange={(e) => setSelectedRouteId(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(all routes)</option>
            {availableOptions.routes.map(routeId => (
              <option key={routeId} value={routeId}>
                Route {globalData.routes?.[routeId]?.route_short_name || routeId}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block mb-1">Direction</label>
          <select
            value={selectedDirectionId}
            onChange={(e) => setSelectedDirectionId(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(both directions)</option>
            <option value="0">Direction 0</option>
            <option value="1">Direction 1</option>
          </select>
        </div>

        <div>
          <label className="block mb-1">Stop</label>
          <select
            value={selectedStopId}
            onChange={(e) => setSelectedStopId(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(all stops)</option>
            {availableOptions.stops.map(stopId => {
              const stopName = allCombinations.find(c => c.stopId === stopId)?.stopName || stopId
              return (
                <option key={stopId} value={stopId}>
                  {stopName} ({stopId})
                </option>
              )
            })}
          </select>
        </div>

        <div>
          <label className="block mb-1">Time Type</label>
          <select
            value={selectedTimeType}
            onChange={(e) => setSelectedTimeType(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(all time types)</option>
            {globalData.time_types?.map(timeType => (
              <option key={timeType} value={timeType}>
                {timeType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results */}
      <div className="space-y-4">
        {filteredResults.map((result, idx) => (
          <ResultTile 
            key={idx}
            result={result}
            index={idx}
            globalData={globalData}
          />
        ))}

        {filteredResults.length === 0 && (
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

export default ExplorePage