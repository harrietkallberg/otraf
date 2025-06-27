// src/pages/route/index.tsx
import React, { useEffect, useState } from 'react'
import {
  useParams,
  NavLink,
  Routes,
  Route,
  Navigate,
  Link,
} from 'react-router-dom'

import Overview    from './Overview'
import Punctuality from './Punctuality'
import Violations  from './Violations'
import Analytics   from './Analytics'

// Define your tabs in one place
const TABS = [
  { id: 'overview',    label: 'Overview'    },
  { id: 'punctuality', label: 'Punctuality' },
  { id: 'violations',  label: 'Violations'  },
  { id: 'analytics',   label: 'Analytics'   },
]

export default function RouteLayout() {
  const { rid } = useParams<{ rid: string }>()
  const [nav, setNav] = useState<any>(null)

  useEffect(() => {
    if (!rid) return
    fetch(`/api/routes/${rid}/navigation`)
      .then(r => r.json())
      .then(setNav)
      .catch(console.error)
  }, [rid])

  if (!nav) return <div>Loading route…</div>

  return (
    <>
      {/* Breadcrumb + Title */}
      <div className="flex items-center mb-4">
        <Link
          to="/routes"
          className="text-sm text-blue-600 hover:underline mr-4"
        >
          ← All Routes
        </Link>
        <h1 className="text-2xl font-bold">Route {nav.route_name}</h1>
      </div>

      {/* Tab bar */}
      <nav className="flex space-x-4 mb-4 border-b">
        {TABS.map(tab => (
          <NavLink
            key={tab.id}
            to={tab.id}
            end={tab.id === 'overview'}
            className={({ isActive }) =>
              isActive
                ? 'pb-1 border-b-2 font-semibold'
                : 'pb-1 hover:border-b'
            }
          >
            {tab.label}
          </NavLink>
        ))}
      </nav>

      {/* Nested routes */}
      <Routes>
        <Route path="/" element={<Navigate to="overview" replace />} />
        <Route path="overview"    element={<Overview nav={nav} />} />
        <Route path="punctuality" element={<Punctuality nav={nav} />} />
        <Route path="violations"  element={<Violations nav={nav} />} />
        <Route path="analytics"   element={<Analytics nav={nav} />} />
      </Routes>
    </>
  )
}
