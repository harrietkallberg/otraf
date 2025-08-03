import React, { useContext } from 'react'
import { useNavigate } from 'react-router-dom' // Changed from Link to useNavigate
import { useAuth } from '../contexts/AuthContext'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { useStopData } from '../contexts/StopDataContext'

type StopMeta = {
  stop_name: string
  stop_ids: []
}

export default function StopsList() {
  const globalData = useContext(GlobalDataContext)
  const { user, session } = useAuth()
  const { setParentId } = useStopData()
  const navigate = useNavigate() // Add this line
  console.log('StopsList rendering, user:', !!user, 'session:', !!session)

  if (!globalData) {
    return <div className="px-6 py-4">Loading stops...</div>
  }

  const list = Object.entries(globalData.stops).map(([id, meta]) => ({
    id,
    stopName: meta.stop_name,
    stopIds: meta.stop_ids
  })) // Fixed missing closing parenthesis

  const handleStopSelect = (parentId: string) => {
    console.log('Setting parentId in context:', parentId)
    setParentId(parentId)
    
    // Then navigate to the route AFTER context is updated
    navigate(`/stops/${parentId}`)
  }

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">All Stops</h2>
      <div className="space-y-4">
        {list.map((r) => (
          <div
            key={r.id}
            className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5 cursor-pointer"
            onClick={() => handleStopSelect(r.id)}
          >
            <div>
              <h3 className="text-lg font-medium">{r.stopName}</h3>
              <div className="text-sm text-gray-500 mt-2 space-y-2">
                <div>
                  <span className="font-medium">Parent Station:</span> {r.id}
                </div>
                <div>
                  <div className="font-medium">Stop IDs:</div>
                  {Array.isArray(r.stopIds) ? (
                    <ul className="list-disc list-inside mt-1 space-y-1">
                      {r.stopIds.map((id, index) => (
                        <li key={index}>{id}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="mt-1">{r.stopIds}</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}