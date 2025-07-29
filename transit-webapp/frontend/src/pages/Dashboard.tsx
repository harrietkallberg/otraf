// src/pages/Dashboard.tsx
import React, { useContext, useEffect, useState } from 'react'
import { GlobalDataContext, ViolationEntry } from '../contexts/GlobalDataContext'
import { useAuth } from '../contexts/AuthContext'


type RoutesIndex = Record<
  string,
  { route_long_name: string; route_short_name: string }
>

const Dashboard: React.FC = () => {
  const globalData = useContext(GlobalDataContext)
  const [routes, setRoutes] = useState<RoutesIndex | null>(null)
  const { user } = useAuth()
  // fetch global routes
  useEffect(() => {
    if (!user) return
    fetch('/api/global/routes', {
          headers: { 'X-User-Id': user.id }
        })
      .then((r) => r.json())
      .then((json: RoutesIndex) => setRoutes(json))
      .catch(console.error)
  }, [user])

  if (!globalData || routes === null) {
    return <div className="p-6">Loading dashboard…</div>
  }

  const { labels, violations, time_types, stops } = globalData

  const totalStops = Object.keys(stops).length
  const totalLabels = Object.keys(labels).length
  const totalViolations = Object.values(violations).length
  const totalRoutes = Object.keys(routes).length
  const totalTimeTypes = time_types.length

  // breakdown of violations by severity
  const severityCounts = Object.values(violations).reduce<Record<string, number>>((acc, v) => {
    const sev = String(v.severity)
    acc[sev] = (acc[sev] || 0) + 1
    return acc
  }, {})

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card title="Routes" value={totalRoutes} />
        <Card title="Stops" value={totalStops} />
        <Card title="Labels" value={totalLabels} />
        <Card title="Violations" value={totalViolations} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow p-5">
          <h2 className="text-xl font-semibold mb-4">Violations by Severity</h2>
          <ul className="space-y-1">
            {Object.entries(severityCounts)
              .sort((a, b) => Number(a[0]) - Number(b[0]))
              .map(([sev, count]) => (
                <li key={sev} className="flex justify-between">
                  <span>Severity {sev}</span>
                  <span className="font-semibold">{count}</span>
                </li>
              ))}
          </ul>
        </div>
        <div className="bg-white rounded-2xl shadow p-5">
          <h2 className="text-xl font-semibold mb-4">Other Summary</h2>
          <p>
            <strong>Time Types:</strong> {time_types.join(', ')}
          </p>
        </div>
      </div>
    </div>
  )
}

interface CardProps {
  title: string
  value: number
}

const Card: React.FC<CardProps> = ({ title, value }) => (
  <div className="bg-white rounded-2xl shadow p-5 flex flex-col items-center">
    <h2 className="text-lg font-medium">{title}</h2>
    <p className="text-3xl font-bold mt-2">{value}</p>
  </div>
)

export default Dashboard


