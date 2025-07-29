import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

type RouteMeta = {
  route_long_name: string
  route_short_name: string
}
type Violation = { route_id: string }

export default function RoutesList() {
  const [routes, setRoutes] = useState<Record<string, RouteMeta>>({})
  const [violations, setViolations] = useState<Violation[]>([])
  const { user } = useAuth()

  useEffect(() => {
    if (!user) return
    Promise.all([
      fetch('/api/global/routes', {
        headers: { 'X-User-Id': user.id }
      }).then(r => r.json()),
      fetch('/api/global/violations', {
        headers: { 'X-User-Id': user.id }
      }).then(r => r.json()),
    ])
      .then(([routesJson, violJson]) => {
        setRoutes(routesJson)
        setViolations(violJson)
      })
      .catch(console.error)
  }, [user])

  const list = Object.entries(routes).map(([id, meta]) => ({
    id,
    longName: meta.route_long_name,
    shortName: meta.route_short_name,
    hasViol: violations.some(v => v.route_id === id),
  }))

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">All Routes</h2>
      <div className="space-y-4">
        {list.map((r) => (
          <Link
            key={r.id}
            to={`/routes/${r.id}`}
            className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5 flex justify-between items-center"
          >
            <div>
              <h3 className="text-lg font-medium">Route {r.longName}</h3>
              <p className="text-sm text-gray-500">
                Short name: {r.shortName}
              </p>
            </div>
            <span
              className={`w-3 h-3 rounded-full ${r.hasViol ? 'bg-red-500' : 'bg-green-500'}`}
              aria-label={r.hasViol ? 'Has violations' : 'No violations'}
            />
          </Link>
        ))}
      </div>
    </div>
  )
}
