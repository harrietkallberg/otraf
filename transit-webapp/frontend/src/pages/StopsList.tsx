import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useStopData } from '../contexts/StopDataContext'
import { useAuth } from '../contexts/AuthContext'

type Violation = { stop_id: string }

interface AggregatedStop {
  stopName: string
  stopIds: string[]
  routes: string[]  // Add routes to the interface
  hasViol: boolean
}

export default function StopsList() {
  const [stops, setStops] = useState<Record<string, any>>({})
  const [violations, setViolations] = useState<Violation[]>([])
  const { user, session } = useAuth()
  const { setParentId } = useStopData() // Access stop context to set selected stop name

  useEffect(() => {
    if (!user || !session?.access_token) return

    const headers = { 
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,
    }

    // Fetching stops and violations data
    Promise.all([
      fetch('/api/global/stops', { headers }).then(r => r.json()),
      fetch('/api/global/violations', { headers }).then(r => r.json()),
    ])
      .then(([stopsJson, violJson]) => {
        setStops(stopsJson)
        setViolations(violJson)
      })
      .catch(console.error)
  }, [user, session])

  const handleStopSelect = (parentId: string) => {
    setParentId(parentId) // Set stop name in context
  }

  // Group stops by stop name, assuming that each stop name can have multiple stop ids
  const aggregatedStops: Record<string, AggregatedStop> = Object.entries(stops).reduce((acc: Record<string, AggregatedStop>, [id, meta]) => {
    const parentId = meta.parentId

    if (!acc[parentId]) {
      acc[parentId] = {
        stopName: meta.name,
        stopIds: [],
        routes: [],  // Initialize routes array
        hasViol: false,
      }
    }

    // Add the current stopId to the aggregated stop's stopIds
    acc[parentId].stopIds.push(id)

    // Add the current route to the aggregated stop's routes
    meta.routes.forEach((route: string) => {
      if (!acc[parentId].routes.includes(route)) {
        acc[parentId].routes.push(route)
      }
    })

    // Check if any of the stop_ids in the current stop have violations
    const hasViol = violations.some((v) => acc[parentId].stopIds.includes(v.stop_id))
    acc[parentId].hasViol = acc[parentId].hasViol || hasViol

    return acc
  }, {})
  
  // Convert aggregatedStops object back to an array for rendering
  const list = Object.values(aggregatedStops)

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">All Stops</h2>
      <div className="space-y-4">
        {list.map((stop) => (
          <Link
            key={stop.stopName}
            to={`/stops/${stop.stopName}`} // Navigate using stop name
            className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5"
            onClick={() => handleStopSelect(stop.stopName)} // Pass stop name to context
          >
            <div className="flex justify-between items-center">
              <span className="text-lg">{stop.stopName}</span>
              <span
                className={`w-3 h-3 rounded-full ${stop.hasViol ? 'bg-red-500' : 'bg-green-500'}`}
                aria-label={stop.hasViol ? 'Has violations' : 'No violations'}
              />
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
