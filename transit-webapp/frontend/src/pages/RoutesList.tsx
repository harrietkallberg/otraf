import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

export default function RoutesList() {
  const [routes, setRoutes] = useState<Record<string, any>>({})

  useEffect(() => {
    // now we're not returning anything from this callback
    fetch('/api/routes')
      .then(r => r.json())
      .then(setRoutes)
      .catch(err => {
        console.error('Failed to load routes', err)
      })
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">All Routes</h1>
      <ul className="space-y-2">
        {Object.entries(routes).map(([rid, info]) => (
          <li key={rid}>
            <Link to={`/routes/${rid}`} className="text-blue-600 hover:underline">
              {info.route_short_name} — {info.route_long_name}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
