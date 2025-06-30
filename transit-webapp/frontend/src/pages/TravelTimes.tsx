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
  const [fromStop, setFromStop] = useState('')
  const [toStop, setToStop] = useState('')
  const [timeType, setTimeType] = useState('')

  useEffect(() => {
    Promise.all([
      fetch('/api/global/stops').then(r => r.json()),
      fetch('/api/global/travel_times').then(r => r.json())
    ]).then(([stops, times]) => {
      setStopIndex(stops)
      setSegments(times)
    })
  }, [])

  const allFromStops = new Set(segments.map(s => s.from_stop_id))
  const allToStops = new Set(segments.map(s => s.to_stop_id))

  const fromStopOptions = Array.from(
    new Set(
      segments
        .filter(s => !toStop || s.to_stop_id === toStop)
        .map(s => s.from_stop_id)
    )
  )
    .map(id => ({ id, name: stopIndex[id]?.stop_name || id }))
    .sort((a, b) => a.name.localeCompare(b.name))

  const toStopOptions = Array.from(
    new Set(
      segments
        .filter(s => !fromStop || s.from_stop_id === fromStop)
        .map(s => s.to_stop_id)
    )
  )
    .map(id => ({ id, name: stopIndex[id]?.stop_name || id }))
    .sort((a, b) => a.name.localeCompare(b.name))

  const filtered = segments.filter(s =>
    (!fromStop || s.from_stop_id === fromStop) &&
    (!toStop || s.to_stop_id === toStop) &&
    (!timeType || s.time_type === timeType) &&
    (typeof s.aggregated?.mean === 'number' || s.by_route?.some(r => typeof r.mean === 'number'))
  )

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">🕒 Search Travel Times</h1>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <label>From Stop</label>
          <select
            value={fromStop}
            onChange={e => setFromStop(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(any)</option>
            {fromStopOptions.map(({ id, name }) => (
              <option key={id} value={id}>
                {name} ({id})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label>To Stop</label>
          <select
            value={toStop}
            onChange={e => setToStop(e.target.value)}
            className="w-full border p-2 rounded"
          >
            <option value="">(any)</option>
            {toStopOptions.map(({ id, name }) => (
              <option key={id} value={id}>
                {name} ({id})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label>Time Type</label>
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

      <div className="space-y-3">
        {filtered.map((s, idx) => (
          <div key={idx} className="border p-4 rounded shadow-sm bg-white">
            <div className="text-sm text-gray-500 mb-1">
              {s.time_type.toUpperCase()}
            </div>
            <div className="text-lg font-semibold">
              {s.from_stop_name} → {s.to_stop_name}
            </div>

            <div className="mt-1 text-sm text-gray-700">
              {typeof s.aggregated?.mean === 'number' ? (
                <div>
                  Aggregated: <strong>{s.aggregated.mean.toFixed(1)}</strong> seconds
                </div>
              ) : (
                <div>No aggregated value available</div>
              )}

              {s.by_route?.length > 0 && (
                <div className="mt-1">
                  <div className="font-medium text-gray-500">Per Route:</div>
                  <ul className="list-disc list-inside text-sm">
                    {s.by_route.map((r, i) => (
                      <li key={i}>
                        Route {r.route_id}, dir {r.direction_id}:{' '}
                        {typeof r.mean === 'number' ? `${r.mean.toFixed(1)} sec` : 'n/a'}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <p className="text-gray-500 mt-4">No matching travel time segments found.</p>
        )}
      </div>
    </div>
  )
}

export default TravelTimes
