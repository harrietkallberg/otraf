// src/pages/route/Punctuality.tsx
import React, { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from 'recharts'

interface Direction {
  direction_id: string
  direction_name?: string
}

interface Nav {
  route_id: string
  route_short_name: string
  route_long_name: string
  directions: Direction[]
}

interface PerfRecord {
  stop_id: string
  stop_name: string
  on_time_pct: number
  early_pct?: number
  late_pct?: number
  [key: string]: any
}

interface Props {
  nav: Nav
}

export default function Punctuality({ nav }: Props) {
  // Map of direction → array of performance records
  const [perfData, setPerfData] = useState<Record<string, PerfRecord[]>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!nav?.directions) return

    // Fetch performance for each direction in parallel
    const fetchAll = async () => {
      setLoading(true)
      try {
        const dirIds = nav.directions.map(d => d.direction_id)
        const results = await Promise.all(
          dirIds.map(dir =>
            fetch(`/api/routes/${nav.route_id}/directions/${dir}/performance`)
              .then(res => res.json())
          )
        )
        // Build a map: { [dir]: data[] }
        const map: Record<string, PerfRecord[]> = {}
        dirIds.forEach((dir, idx) => {
          map[dir] = results[idx]
        })
        setPerfData(map)
      } catch (err) {
        console.error("Failed to load performance data", err)
      } finally {
        setLoading(false)
      }
    }

    fetchAll()
  }, [nav])

  if (loading) {
    return <div>Loading punctuality charts…</div>
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">
        Punctuality Flow for {nav.route_short_name} — {nav.route_long_name}
      </h2>

      {nav.directions.map(dir => {
        const data = perfData[dir.direction_id] || []
        return (
          <section key={dir.direction_id} className="mb-8">
            <h3 className="text-xl font-semibold mb-2">
              Direction {dir.direction_name ?? dir.direction_id}
            </h3>

            {data.length === 0 ? (
              <div>No data for this direction.</div>
            ) : (
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 50 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="stop_name" 
                      interval={0} 
                      angle={-45} 
                      textAnchor="end" 
                      height={60}
                    />
                    <YAxis 
                      domain={[0, 100]} 
                      label={{ value: "% On Time", angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip formatter={(value: any) => `${value}%`} />
                    <Legend verticalAlign="top" height={36}/>
                    <Bar 
                      dataKey="on_time_pct" 
                      name="% On Time" 
                      barSize={20} 
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}
