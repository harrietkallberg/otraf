// src/pages/StopsList.tsx
import React, { useContext } from 'react'
import { Link } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'

interface StopGroup {
  name: string
  ids: string[]
  hasViolation: boolean
}

const StopsList: React.FC = () => {
  const globalData = useContext(GlobalDataContext)!
  const stopsIndex = globalData.stops   // Record<stop_id, { stop_name, violation_stats, … }>

  // Group by stop_name
  const groupsMap: Record<string, StopGroup> = {}
  Object.entries(stopsIndex).forEach(([id, entry]) => {
    const name = entry.stop_name
    // check across all domains for any violations
    const hasViol = Object.values(entry.violation_stats).some(
      (dom: any) => dom.occurrences > 0
    )

    if (!groupsMap[name]) {
      groupsMap[name] = { name, ids: [], hasViolation: false }
    }
    groupsMap[name].ids.push(id)
    if (hasViol) groupsMap[name].hasViolation = true
  })

  // Turn into a sorted array
  const groups = Object.values(groupsMap).sort((a, b) =>
    a.name.localeCompare(b.name)
  )

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">All Stops</h2>
      <div className="space-y-4">
        {groups.map((g) => (
          <Link
            key={g.name}
            to={`/stops/${g.ids[0]}`}
            className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5 flex justify-between items-center"
          >
            <div>
              <h3 className="text-lg font-medium">{g.name}</h3>
              <p className="text-sm text-gray-500">
                {g.ids.length > 1
                  ? `${g.ids.length} stop IDs`
                  : `ID: ${g.ids[0]}`}
              </p>
            </div>
            <span
              className={`w-3 h-3 rounded-full ${
                g.hasViolation ? 'bg-red-500' : 'bg-green-500'
              }`}
              aria-label={g.hasViolation ? 'Has violations' : 'No violations'}
            />
          </Link>
        ))}
      </div>
    </div>
  )
}

export default StopsList
