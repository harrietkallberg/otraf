// src/pages/route/Overview.tsx
import React from 'react'
import { Link, useParams } from 'react-router-dom'

interface Direction {
  direction_id: string
  direction_label: string
}

interface Nav {
  route_id: string
  route_name: string
  directions: Direction[]
}

interface Props {
  nav: Nav
}

const Overview: React.FC<Props> = ({ nav }) => {
  const { rid } = useParams<{ rid: string }>()

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">
        Directions for Route {nav.route_name}
      </h2>
      <ul className="space-y-2">
        {nav.directions.map((dir: Direction) => (
          <li key={dir.direction_id}>
            <Link
              to={`/routes/${rid}/${dir.direction_id}`}
              className="text-blue-600 hover:underline"
            >
              {dir.direction_label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default Overview
