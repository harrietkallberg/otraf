// src/pages/TravelTimes.tsx
import React, { useEffect, useState, useMemo } from 'react'
import { useAuth } from '../contexts/AuthContext'

interface ByRouteEntry {
  route_id: number
  direction_id: string
  mean: number | null
  sample_size: number
}

interface TravelSegment {
  seg_key: string
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

interface ParsedSegmentKey {
  route_id: number
  direction_id: string
  from_stop_id: string
  to_stop_id: string
  time_type: string
}

const TravelTimes: React.FC = () => {
  const [segments, setSegments] = useState<TravelSegment[]>([])
  const [stopIndex, setStopIndex] = useState<Record<string, any>>({})
  const [route, setRoute] = useState('')
  const [fromName, setFromName] = useState('')
  const [toName, setToName] = useState('')
  const [timeType, setTimeType] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [expandedSegments, setExpandedSegments] = useState<Set<number>>(new Set())
  const { user, session } = useAuth()
  
  // No longer need to parse seg_keys since we have clean array format

  // filter out purely numeric names
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

  useEffect(() => {
    if (!user || !session?.access_token) {
      setLoading(false)
      return
    }
    
    setLoading(true)
    setError(null)
    
    const headers = {
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,
    }
    
    Promise.all([
      fetch('/api/global/stops', { headers }).then(async r => {
        if (!r.ok) {
          throw new Error(`Failed to fetch stops: ${r.status} ${r.statusText}`)
        }
        return r.json()
      }),
      fetch('/api/global/travel_times', { headers }).then(async r => {
        if (!r.ok) {
          throw new Error(`Failed to fetch travel times: ${r.status} ${r.statusText}`)
        }
        return r.json()
      })
    ])
    .then(([stops, timesData]) => {
      console.log('Stops data:', stops)
      console.log('Times data keys:', Object.keys(timesData).length)
      
      if (!stops || typeof stops !== 'object') {
        throw new Error('Invalid stops data format')
      }
      
      let timesArray: any[] = []
      
      if (Array.isArray(timesData)) {
        timesArray = timesData
      } else if (timesData && typeof timesData === 'object') {
        timesArray = Object.values(timesData)
        console.log(`Converted dictionary to array: ${timesArray.length} segments`)
      } else {
        throw new Error(`Expected travel times to be an object or array, got: ${typeof timesData}`)
      }
      
      setStopIndex(stops)
      const withNames = timesArray.map(s => ({
        ...s,
        from_stop_name: stops[s.from_stop_id]?.stop_name || s.from_stop_id,
        to_stop_name: stops[s.to_stop_id]?.stop_name || s.to_stop_id,
      }))
      setSegments(withNames)
      setLoading(false)
    })
    .catch((err) => {
      console.error('Error loading travel times data:', err)
      setError(err.message || 'Failed to load travel times data')
      setLoading(false)
    })
  }, [user, session])

  // Clear dependent filters when parent filters change
  useEffect(() => {
    setFromName('')
    setToName('')
  }, [route, timeType])

  useEffect(() => {
    setToName('')
  }, [fromName])

  // Efficiently compute available options using direct data access
  const availableOptions = useMemo(() => {
    const routes = new Set<number>()
    const timeTypes = new Set<string>()
    const fromStops = new Set<string>()
    const toStops = new Set<string>()
    const validCombinations = new Set<string>()

    segments.forEach(segment => {
      // Check if segment has actual data
      const hasData = (typeof segment.aggregated?.mean === 'number') ||
                     (segment.by_route?.some(r => typeof r.mean === 'number'))
      
      if (!hasData) return

      const from_stop_name = segment.from_stop_name
      const to_stop_name = segment.to_stop_name
      const time_type = segment.time_type

      // Get all routes for this segment
      const segmentRoutes = segment.by_route?.map(r => r.route_id) || []
      
      segmentRoutes.forEach(route_id => {
        const combKey = `${route_id}|${from_stop_name}|${to_stop_name}|${time_type}`
        validCombinations.add(combKey)
        routes.add(route_id)
      })

      timeTypes.add(time_type)
      if (hasLetters(from_stop_name)) fromStops.add(from_stop_name)
      if (hasLetters(to_stop_name)) toStops.add(to_stop_name)
    })

    // Filter options based on current selections
    const getFilteredRoutes = () => {
      if (!fromName && !toName && !timeType) return Array.from(routes).sort((a, b) => a - b)
      
      const filtered = new Set<number>()
      validCombinations.forEach(combKey => {
        const [r, f, t, tt] = combKey.split('|')
        if ((!fromName || f === fromName) &&
            (!toName || t === toName) &&
            (!timeType || tt === timeType)) {
          filtered.add(Number(r))
        }
      })
      return Array.from(filtered).sort((a, b) => a - b)
    }

    const getFilteredTimeTypes = () => {
      if (!route && !fromName && !toName) return Array.from(timeTypes).sort()
      
      const filtered = new Set<string>()
      validCombinations.forEach(combKey => {
        const [r, f, t, tt] = combKey.split('|')
        if ((!route || Number(r) === Number(route)) &&
            (!fromName || f === fromName) &&
            (!toName || t === toName)) {
          filtered.add(tt)
        }
      })
      return Array.from(filtered).sort()
    }

    const getFilteredFromStops = () => {
      if (!route && !toName && !timeType) {
        return Array.from(fromStops).sort((a, b) => a.localeCompare(b))
      }
      
      const filtered = new Set<string>()
      validCombinations.forEach(combKey => {
        const [r, f, t, tt] = combKey.split('|')
        if ((!route || Number(r) === Number(route)) &&
            (!toName || t === toName) &&
            (!timeType || tt === timeType) &&
            hasLetters(f)) {
          filtered.add(f)
        }
      })
      return Array.from(filtered).sort((a, b) => a.localeCompare(b))
    }

    const getFilteredToStops = () => {
      if (!route && !fromName && !timeType) {
        return Array.from(toStops).sort((a, b) => a.localeCompare(b))
      }
      
      const filtered = new Set<string>()
      validCombinations.forEach(combKey => {
        const [r, f, t, tt] = combKey.split('|')
        if ((!route || Number(r) === Number(route)) &&
            (!fromName || f === fromName) &&
            (!timeType || tt === timeType) &&
            hasLetters(t)) {
          filtered.add(t)
        }
      })
      return Array.from(filtered).sort((a, b) => a.localeCompare(b))
    }

    return {
      routes: getFilteredRoutes(),
      timeTypes: getFilteredTimeTypes(),
      fromStops: getFilteredFromStops(),
      toStops: getFilteredToStops(),
      validCombinations
    }
  }, [segments, route, fromName, toName, timeType])

  // Filter segments for display
  const filteredSegments = useMemo(() => {
    return segments.filter(segment => {
      const routeMatch = !route || segment.by_route?.some(r => r.route_id === Number(route))
      const fromMatch = !fromName || segment.from_stop_name === fromName
      const toMatch = !toName || segment.to_stop_name === toName
      const timeMatch = !timeType || segment.time_type === timeType
      
      // Check if segment has actual data for the selected route
      const hasData = !route 
        ? (typeof segment.aggregated?.mean === 'number') ||
          (segment.by_route?.some(r => typeof r.mean === 'number'))
        : segment.by_route?.some(r => r.route_id === Number(route) && typeof r.mean === 'number')
      
      return routeMatch && fromMatch && toMatch && timeMatch && hasData
    })
  }, [segments, route, fromName, toName, timeType])

  if (loading) {
    return <div className="p-6">Loading travel times...</div>
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong>Error:</strong> {error}
        </div>
        <button 
          onClick={() => window.location.reload()} 
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">🕒 Search Travel Times</h1>
      
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
              <option key={rid} value={rid}>Route {rid}</option>
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

        <div>
          <label className="block mb-1">From Stop</label>
          <select
            value={fromName}
            onChange={e => setFromName(e.target.value)}
            className="w-full border p-2 rounded"
            disabled={availableOptions.fromStops.length === 0}
          >
            <option value="">(any stop)</option>
            {availableOptions.fromStops.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          {availableOptions.fromStops.length === 0 && (
            <div className="text-xs text-gray-500 mt-1">No stops available for current filters</div>
          )}
        </div>

        <div>
          <label className="block mb-1">To Stop</label>
          <select
            value={toName}
            onChange={e => setToName(e.target.value)}
            className="w-full border p-2 rounded"
            disabled={availableOptions.toStops.length === 0}
          >
            <option value="">(any stop)</option>
            {availableOptions.toStops.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          {availableOptions.toStops.length === 0 && (
            <div className="text-xs text-gray-500 mt-1">No stops available for current filters</div>
          )}
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
                <div className="space-y-1">
                  {s.by_route
                    ?.filter(r2 => r2.route_id === Number(route))
                    .map((r2, i) => (
                      <div key={i}>
                        Route {r2.route_id}:{' '}
                        {typeof r2.mean === 'number'
                          ? <strong>{r2.mean.toFixed(1)} s</strong>
                          : 'n/a'}
                        {r2.sample_size && (
                          <span className="text-gray-500 ml-2">
                            ({r2.sample_size.toLocaleString()} samples)
                          </span>
                        )}
                      </div>
                    ))
                  }
                </div>
              ) : (
                <div className="space-y-2">
                  <div>
                    Aggregated:{' '}
                    {typeof s.aggregated?.mean === 'number'
                      ? <strong>{s.aggregated.mean.toFixed(1)} s</strong>
                      : 'n/a'}
                    {s.aggregated?.sample_size && (
                      <span className="text-gray-500 ml-2">
                        ({s.aggregated.sample_size.toLocaleString()} samples)
                      </span>
                    )}
                  </div>
                  
                  <button
                    onClick={() => toggleSegmentDetails(idx)}
                    className="text-blue-600 hover:text-blue-800 text-sm underline"
                  >
                    {expandedSegments.has(idx) ? 'Hide Details' : 'Show Details'}
                  </button>
                  
                  {expandedSegments.has(idx) && (
                    <div className="mt-2 pl-4 border-l-2 border-gray-200 space-y-1">
                      {s.by_route?.map((r2, i) => (
                        <div key={i} className="text-xs">
                          Route {r2.route_id}, dir {r2.direction_id}:{' '}
                          {typeof r2.mean === 'number'
                            ? `${r2.mean.toFixed(1)} s`
                            : 'n/a'}
                          {r2.sample_size && (
                            <span className="text-gray-500 ml-2">
                              ({r2.sample_size.toLocaleString()} samples)
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
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