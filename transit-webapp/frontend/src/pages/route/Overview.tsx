// src/pages/route/Overview.tsx
import React, { useContext } from 'react'
import { Link, useParams } from 'react-router-dom'
import { GlobalDataContext } from '../../contexts/GlobalDataContext'

interface DirectionNav {
  direction_id: string
  direction_label_keys: string[]
  has_direction_violations: boolean
  direction_violation_keys: string[]
}

interface Nav {
  route_id: string
  route_name: string
  directions: DirectionNav[]
}

interface Props {
  nav: Nav
}

const Overview: React.FC<Props> = ({ nav }) => {
  const { rid } = useParams<{ rid: string }>()
  const global = useContext(GlobalDataContext)
  const { labels, violations } = global!

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">
        Directions for Route {nav.route_name}
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {nav.directions.map(dir => {
          // pick the first label key for description fallback
          const labelKey = dir.direction_label_keys[0]
          const label =
            labels[labelKey]?.description ??
            `Direction ${dir.direction_id}`

          // determine tooltip text & color
          const hasViol = dir.has_direction_violations
          const statusColor = hasViol ? 'bg-red-500' : 'bg-green-500'
          const statusLabel = hasViol ? 'Has violations' : 'No violations'

          return (
            <Link
              key={dir.direction_id}
              to={`/routes/${rid}/${dir.direction_id}`}
              className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5 relative group"
            >
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-medium text-gray-900">
                  {label}
                </h3>
                <span
                  className={`ml-2 inline-block w-3 h-3 ${statusColor} rounded-full`}
                  title={statusLabel}
                  aria-label={statusLabel}
                />
              </div>
              <div className="mt-2 text-sm text-gray-500">
                ID: {dir.direction_id}
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}

export default Overview
