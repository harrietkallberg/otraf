// src/pages/TravelTimes.tsx
import React, { useEffect, useState } from 'react'

interface ByRouteEntry {
  route_id: string
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
  const [stopIndex, setStopIndex] = useState<Record<string, any>>({})
  const [route, setRoute]       = useState('')
  const [fromName, setFromName] = useState('')
  const [toName, setToName]     = useState('')
  const [timeType, setTimeType] = useState('')

  // filter out purely numeric names
  const hasLetters = (s: string) => /\D/.test(s)

  useEffect(() => {
    Promise.all([
      fetch('/api/global/stops').then(r => r.json()),
      fetch('/api/global/travel_times').then(r => r.json())
    ]).then(([stops, times]: [Record<string, any>, Omit<TravelSegment, 'from_stop_name' | 'to_stop_name'>[]]) => {
      setStopIndex(stops)
      // attach names
      const withNames = times.map(s => ({
        ...s,
        from_stop_name: stops[s.from_stop_id]?.stop_name || s.from_stop_id,
        to_stop_name:   stops[s.to_stop_id]?.stop_name   || s.to_stop_id,
      }))
      setSegments(withNames)
    })
  }, [])

  // build name→IDs map
  const nameToIds: Record<string, string[]> = {}
  segments.forEach(s => {
    nameToIds[s.from_stop_name] ||= []
    nameToIds[s.to_stop_name]   ||= []
    nameToIds[s.from_stop_name].push(s.from_stop_id)
    nameToIds[s.to_stop_name].push(s.to_stop_id)
  })
  Object.keys(nameToIds)
    .forEach(n => nameToIds[n] = Array.from(new Set(nameToIds[n])))

  // get all route IDs
  const allRoutes = Array.from(
    new Set(segments.flatMap(s => s.by_route.map(r => r.route_id)))
  ).sort()

  // options for From/To drop-downs, filtered by route/timeType/other
  const fromOptions = Array.from(new Set(
    segments
      .filter(s =>
        (!route   || s.by_route.some(r2 => r2.route_id === route)) &&
        (!toName  || s.to_stop_name === toName) &&
        (!timeType|| s.time_type === timeType)
      )
      .map(s => s.from_stop_name)
  ))
    .filter(hasLetters)
    .sort((a, b) => a.localeCompare(b))

  const toOptions = Array.from(new Set(
    segments
      .filter(s =>
        (!route   || s.by_route.some(r2 => r2.route_id === route)) &&
        (!fromName|| s.from_stop_name === fromName) &&
        (!timeType|| s.time_type === timeType)
      )
      .map(s => s.to_stop_name)
  ))
    .filter(hasLetters)
    .sort((a, b) => a.localeCompare(b))

  // finally filter which segments to display
  const filtered = segments.filter(s => {
    const byThisRoute = !route || s.by_route.some(r2 => r2.route_id === route)
    const fromMatch   = !fromName || nameToIds[fromName].includes(s.from_stop_id)
    const toMatch     = !toName   || nameToIds[toName].includes(s.to_stop_id)
    const timeMatch   = !timeType || s.time_type === timeType
    const hasAnyValue = 
      (!route && typeof s.aggregated.mean === 'number') ||
      ( route &&     s.by_route.some(r2 => r2.route_id === route && typeof r2.mean === 'number'))
    return byThisRoute && fromMatch && toMatch && timeMatch && hasAnyValue
  })

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">🕒 Search Travel Times</h1>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block mb-1">Route</label>
          <select
            value={route}
            onChange={e => setRoute(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(all)</option>
            {allRoutes.map(rid => (
              <option key={rid} value={rid}>Route {rid}</option>
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
            <option value="">(any)</option>
            {fromOptions.map(name => (
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
            <option value="">(any)</option>
            {toOptions.map(name => (
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
            <option value="">(any)</option>
            <option value="am_rush">AM Rush</option>
            <option value="day">Day</option>
            <option value="pm_rush">PM Rush</option>
            <option value="night">Night</option>
            <option value="weekend">Weekend</option>
          </select>
        </div>
      </div>

      <div className="space-y-4">
        {filtered.map((s, idx) => (
          <div key={idx} className="border p-4 rounded shadow-sm bg-white">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">
                {s.time_type.toUpperCase()}
              </span>
              {route ? (
                <span className="text-sm text-blue-600">
                  Route {route}
                </span>
              ) : null}
            </div>

            <div className="text-lg font-semibold my-1">
              {s.from_stop_name} → {s.to_stop_name}
            </div>

            <div className="text-sm text-gray-700">
              {route ? (
                <div className="space-y-1">
                  <div>
                    Aggregated:{' '}
                    {typeof s.aggregated.mean === 'number'
                      ? <strong>{s.aggregated.mean.toFixed(1)} s</strong>
                      : 'n/a'}
                  </div>
                  {s.by_route
                    .filter(r2 => r2.route_id === route)
                    .map((r2, i) => (
                      <div key={i}>
                        Route {r2.route_id}, dir {r2.direction_id}:{' '}
                        {typeof r2.mean === 'number'
                          ? `${r2.mean.toFixed(1)} s`
                          : 'n/a'}
                      </div>
                    ))
                  }
                </div>
              ) : (
                <div>
                  Aggregated:{' '}
                  {typeof s.aggregated.mean === 'number'
                    ? <strong>{s.aggregated.mean.toFixed(1)} s</strong>
                    : 'n/a'}
                </div>
              )}
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <p className="text-gray-500">No matching segments found.</p>
        )}
      </div>
    </div>
  )
}

export default TravelTimes
