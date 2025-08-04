// src/pages/TravelTimes.tsx
import React, { useEffect, useState, useMemo, useContext } from 'react'
import { GlobalDataContext  } from '../contexts/GlobalDataContext'

interface ByRouteEntry {
  route_id: number
  direction_id: string
  mean: number | null
  sample_size: number
}

interface TravelSegment {
  from_stop_id: string
  to_stop_id: string
  from_stop_name: string
  to_stop_name: string
  time_type: string
  aggregated: {
    mean: number | null
    sample_size: number
  }
  by_route: ByRouteEntry[]
}

const TravelTimes: React.FC = () => {
  const [segments, setSegments] = useState<TravelSegment[]>([])
  const [route, setRoute] = useState('')
  const [fromName, setFromName] = useState('')
  const [toName, setToName] = useState('')
  const [timeType, setTimeType] = useState('')
  const [expandedSegments, setExpandedSegments] = useState<Set<number>>(new Set())
  const globalData = useContext(GlobalDataContext)

  // Set segments directly from travel_times when globalData is available
  useEffect(() => {
    if (!globalData?.travel_times) return

    
    const { stops, routes, labels, violations, time_types, travel_times, performance } = globalData
    console.log('Setting travel times data:', travel_times)
    
    setSegments(travel_times)
  }, [globalData])

  const hasLetters = (s: string) => /\D/.test(s)

  const toggleSegmentDetails = (index: number) => {
    const newExpanded = new Set(expandedSegments)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSegments(newExpanded)
  }

  // Smart filter clearing - only clear if current selection becomes invalid
  useEffect(() => {
    if (!fromName || !toName || !Array.isArray(segments)) return // Add array check
    
    // Check if current fromName is still valid
    const validFromStops = new Set<string>()
    const validToStops = new Set<string>()
    
    segments.forEach(segment => {
      const hasData = (typeof segment.aggregated?.mean === 'number') ||
                     (segment.by_route?.some(r => typeof r.mean === 'number'))
      if (!hasData) return

      const routeMatch = !route || segment.by_route?.some(r => r.route_id === Number(route))
      const timeMatch = !timeType || segment.time_type === timeType

      if (routeMatch && timeMatch) {
        if (hasLetters(segment.from_stop_name)) validFromStops.add(segment.from_stop_name)
        if (hasLetters(segment.to_stop_name)) validToStops.add(segment.to_stop_name)
      }
    })

    // Only clear if current selection is no longer valid
    if (fromName && !validFromStops.has(fromName)) {
      setFromName('')
      setToName('') // Also clear toName since fromName changed
    } else if (toName && !validToStops.has(toName)) {
      setToName('')
    }
  }, [route, timeType, fromName, toName, segments])

  // Smart clearing for toName when fromName changes
  useEffect(() => {
    if (!fromName || !toName || !Array.isArray(segments)) return // Add array check
    
    // Check if current toName is still reachable from current fromName
    const validToStops = new Set<string>()
    
    segments.forEach(segment => {
      const hasData = (typeof segment.aggregated?.mean === 'number') ||
                     (segment.by_route?.some(r => typeof r.mean === 'number'))
      if (!hasData) return

      const routeMatch = !route || segment.by_route?.some(r => r.route_id === Number(route))
      const timeMatch = !timeType || segment.time_type === timeType
      const fromMatch = segment.from_stop_name === fromName

      if (routeMatch && timeMatch && fromMatch) {
        if (hasLetters(segment.to_stop_name)) validToStops.add(segment.to_stop_name)
      }
    })

    // Only clear toName if it's no longer reachable from current fromName
    if (toName && !validToStops.has(toName)) {
      setToName('')
    }
  }, [fromName, route, timeType, toName, segments])

  // Much shorter progressive filtering - single pass approach
  const { availableOptions, filteredSegments } = useMemo(() => {
    // Add safety check for segments array
    if (!Array.isArray(segments)) {
      return {
        availableOptions: {
          routes: [],
          timeTypes: [],
          fromStops: [],
          toStops: []
        },
        filteredSegments: []
      }
    }

    const routes = new Set<number>()
    const timeTypes = new Set<string>()
    const fromStops = new Set<string>()
    const toStops = new Set<string>()
    const filtered: TravelSegment[] = []

    segments.forEach(segment => {
      const hasData = (typeof segment.aggregated?.mean === 'number') ||
                     (segment.by_route?.some(r => typeof r.mean === 'number'))
      if (!hasData) return

      // Apply current filters
      const routeMatch = !route || segment.by_route?.some(r => r.route_id === Number(route))
      const fromMatch = !fromName || segment.from_stop_name === fromName
      const toMatch = !toName || segment.to_stop_name === toName
      const timeMatch = !timeType || segment.time_type === timeType

      // If matches all current filters, include in results
      if (routeMatch && fromMatch && toMatch && timeMatch) {
        filtered.push(segment)
      }

      // Collect available options based on partial matches (progressive filtering)
      if (fromMatch && toMatch && timeMatch) segment.by_route?.forEach(r => routes.add(r.route_id))
      if (routeMatch && toMatch && timeMatch) timeTypes.add(segment.time_type)
      if (routeMatch && toMatch && timeMatch && hasLetters(segment.from_stop_name)) fromStops.add(segment.from_stop_name)
      if (routeMatch && fromMatch && timeMatch && hasLetters(segment.to_stop_name)) toStops.add(segment.to_stop_name)
    })

    return {
      availableOptions: {
        routes: Array.from(routes).sort((a, b) => a - b),
        timeTypes: Array.from(timeTypes).sort(),
        fromStops: Array.from(fromStops).sort((a, b) => a.localeCompare(b)),
        toStops: Array.from(toStops).sort((a, b) => a.localeCompare(b))
      },
      filteredSegments: filtered
    }
  }, [segments, route, fromName, toName, timeType])

  if (!globalData) {
    return <div className="p-6">Loading travel times...</div>
  }

  // Add additional safety check
  if (!Array.isArray(globalData.travel_times)) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <h3 className="font-medium">Error Loading Travel Times</h3>
          <p className="text-sm mt-1">Travel times data is not in the expected format.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4"> Search Travel Times</h1>
      
      <div className="mb-4 text-sm text-gray-600">
        Found {segments.length} travel segments total, {filteredSegments.length} matching filters
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block mb-1">Route</label>
          <select
            value={route}
            onChange={e => setRoute(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(all routes)</option>
            {availableOptions.routes.map(rid => (
              <option key={rid} value={rid}>
                Route {globalData.routes?.[rid]?.route_short_name || rid}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block mb-1">From Stop</label>
          <select
            value={fromName}
            onChange={e => setFromName(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(any stop)</option>
            {availableOptions.fromStops.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block mb-1">To Stop</label>
          <select
            value={toName}
            onChange={e => setToName(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(any stop)</option>
            {availableOptions.toStops.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block mb-1">Time Type</label>
          <select
            value={timeType}
            onChange={e => setTimeType(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(any time)</option>
            {availableOptions.timeTypes.map(tt => (
              <option key={tt} value={tt}>
                {tt.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>

      </div>

      <div className="space-y-4">
        {filteredSegments.map((s, idx) => (
          <div key={idx} className="border p-4 rounded shadow-sm bg-white">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">
                {s.time_type?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
              </span>
            </div>

            <div className="text-lg font-semibold my-1">
              {s.from_stop_name} → {s.to_stop_name}
            </div>

            <div className="text-sm text-gray-700">
              {route ? (
                // When a route is selected, show only that route's data - no "Show Details" button
                <div className="space-y-1">
                  {s.by_route
                    ?.filter(r => r.route_id === Number(route))
                    .map((r, i) => (
                      <div key={i}>
                        Route {globalData.routes?.[r.route_id]?.route_short_name || r.route_id}, dir {r.direction_id}:{' '}
                        {typeof r.mean === 'number'
                          ? <strong>{r.mean.toFixed(1)} s</strong>
                          : 'n/a'}
                        {r.sample_size > 0 && (
                          <span className="text-gray-500 ml-2">
                            ({r.sample_size.toLocaleString()} samples)
                          </span>
                        )}
                      </div>
                    ))
                  }
                </div>
              ) : (
                // When no route is selected, show aggregated with expandable details from by_route
                <div className="space-y-2">
                  <div>
                    Aggregated:{' '}
                    {typeof s.aggregated?.mean === 'number'
                      ? <strong>{s.aggregated.mean.toFixed(1)} s</strong>
                      : 'n/a'}
                    {s.aggregated?.sample_size > 0 && (
                      <span className="text-gray-500 ml-2">
                        ({s.aggregated.sample_size.toLocaleString()} samples)
                      </span>
                    )}
                  </div>
                  
                  {s.by_route && s.by_route.length > 0 && (
                    <>
                      <button
                        onClick={() => toggleSegmentDetails(idx)}
                        className="text-blue-600 hover:text-blue-800 text-sm underline"
                      >
                        {expandedSegments.has(idx) ? 'Hide Details' : 'Show Details'}
                      </button>
                      
                      {expandedSegments.has(idx) && (
                        <div className="mt-2 pl-4 border-l-2 border-gray-200 space-y-1">
                          {s.by_route.map((r, i) => (
                            <div key={i} className="text-xs">
                              Route {globalData.routes?.[r.route_id]?.route_short_name || r.route_id}, dir {r.direction_id}:{' '}
                              {typeof r.mean === 'number'
                                ? `${r.mean.toFixed(1)} s`
                                : 'n/a'}
                              {r.sample_size > 0 && (
                                <span className="text-gray-500 ml-2">
                                  ({r.sample_size.toLocaleString()} samples)
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {filteredSegments.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500 text-lg">No matching segments found.</p>
            <p className="text-gray-400 text-sm mt-2">
              Try adjusting your filters to see available travel time data.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default TravelTimes