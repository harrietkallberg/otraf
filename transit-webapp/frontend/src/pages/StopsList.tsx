// src/pages/StopsList.tsx
import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

interface StopMeta {
  stop_name: string
  // add any other fields you want to render
}

export default function StopsList() {
  const [stops, setStops] = useState<Record<string, StopMeta>>({})

  useEffect(() => {
    fetch('/api/stops')
      .then(res => res.json())
      .then(data => setStops(data))
      .catch(console.error)
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">All Stops</h1>
      <ul className="space-y-2">
        {Object.entries(stops).map(([sid, meta]) => (
          <li key={sid}>
            <Link
              to={`/stops/${sid}`}
              className="text-blue-600 hover:underline"
            >
              {meta.stop_name}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
