import React, { useContext } from 'react'
import { useNavigate } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { useRouteData } from '../contexts/RouteDataContext'

type RouteMeta = {
  route_long_name: string
  route_short_name: string
}

export default function RoutesList() {
  const globalData = useContext(GlobalDataContext)
  const { setRouteId } = useRouteData()
  const navigate = useNavigate()

  if (!globalData) {
    return <div className="px-6 py-4">Loading routes...</div>
  }

  const list = Object.entries(globalData.routes).map(([id, meta]) => ({
    id,
    longName: meta.route_long_name,
    shortName: meta.route_short_name,
  }))

  const handleRouteSelect = async (routeId: string) => {
    console.log('Setting routeId in context:', routeId)
    
    // First, set the routeId in context
    setRouteId(routeId)
    
    // Then navigate to the route AFTER context is updated
    navigate(`/routes/${routeId}`)
  }

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">All Routes</h2>
      <div className="space-y-4">
        {list.map((r) => (
          <div
            key={r.id}
            className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5 cursor-pointer"
            onClick={() => handleRouteSelect(r.id)}
          >
            <div>
              <h3 className="text-lg font-medium">Route {r.shortName}</h3>
              <div className="text-sm text-gray-500 mt-2 space-y-2">
                <div>
                  <span className="font-medium">Long Name:</span> {r.longName}
                </div>
                <div>
                  <span className="font-medium">Route ID:</span> {r.id}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}