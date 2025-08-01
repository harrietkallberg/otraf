import React, { useEffect, useState } from 'react'
import { useStopData } from '../contexts/StopDataContext'
import { useAuth } from '../contexts/AuthContext'
import { useParams } from 'react-router-dom'

export default function StopLayout() {
  const { stopName } = useParams<{ stopName: string }>() // Get stopName from the URL
  const { stopData, setStopData } = useStopData() // Access stop context to set/get stop data
  const { user, session } = useAuth() // Access user and session from auth context
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!stopName) {
      setError('Stop name is missing')
      setLoading(false)
      return
    }

    // Fetch stop data when the stopName is selected
    if (stopData && stopData.stop_name === stopName) {
      setLoading(false)
      return
    }

    if (!user || !session?.access_token) {
      setError('User not authenticated')
      setLoading(false)
      return
    }

    setLoading(true)

    const headers = {
      'Authorization': `Bearer ${session.access_token}`,
      'X-User-Id': user.id,
      'X-Refresh-Token': session.refresh_token,
    }

    fetch(`/api/stops/${stopName}`, { headers }) // Fetch stop details using stop name
      .then((res) => res.json())
      .then((data) => {
        setStopData(data)
      })
      .catch((err) => {
        console.error(err)
        setError('Failed to load stop data')
      })
      .finally(() => setLoading(false)) // Set loading to false when data is fetched
  }, [stopName, user, session, stopData, setStopData])

  if (loading) return <div>Loading stop details...</div>
  if (error) return <div>{error}</div>

  return (
    <div>
      <h2>Stop: {stopData?.stop_name}</h2>
      {/* Render the details of the stop */}
      <div>
        <h3>Routes:</h3>
        <ul>
          {stopData?.routes.map((routeId: string) => (
            <li key={routeId}>{routeId}</li>
          ))}
        </ul>
      </div>
      {/* Add more sections here to display other details (e.g., directions, violations, etc.) */}
    </div>
  )
}

