import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom' // To get routeId from URL params
import { useRouteData } from '../contexts/RouteDataContext' // Access context for routeData
import { useAuth } from '../contexts/AuthContext' // Access auth context for user and session

const RouteLayout: React.FC = () => {
  const { routeId } = useParams<{ routeId: string }>() // Get routeId from URL params
  const { routeData, setRouteData } = useRouteData() // Use context to get/set route data
  const { user, session } = useAuth() // Access user and session from auth context
  const [loading, setLoading] = useState<boolean>(false) // Keep loading state until data is available
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    console.log('RouteLayout - useEffect triggered with routeId:', routeId)

    if (!routeId) {
      setError('Route ID is missing')
      setLoading(false)
      return
    }

    // If routeData is already available and corresponds to the correct routeId, no need to fetch again
    if (routeData && routeData.route_id === routeId) {
      console.log('Route data is already available for routeId:', routeId)
      setLoading(false)
      return
    }

    // If routeData doesn't match or is missing, start fetching
    if (!user || !session?.access_token) {
      setError('User not authenticated')
      setLoading(false)
      return
    }

    setLoading(true) // Start loading state

    const headers = {
      'Authorization': `Bearer ${session.access_token}`,
      'X-User-Id': user.id,
      'X-Refresh-Token': session.refresh_token, // Pass the refresh token in a custom header
    }

    console.log('Fetching route data for routeId:', routeId)

    // Fetch the route-specific data
    fetch(`/api/routes/${routeId}/navigation_structure`, { headers })
      .then((res) => res.json())
      .then((data) => {
        if (data) {
          setRouteData(data) // Store the fetched data in context
        } else {
          setError('No data found for the selected route')
        }
      })
      .catch((err) => {
        console.error(err)
        setError('Failed to load route data')
      })
      .finally(() => setLoading(false)) // Set loading to false once the fetch completes
  }, [routeId, user, session, setRouteData]) // Re-run only when routeId, user, or session changes

  if (loading) return <div>Loading route details...</div> // Show loading state while waiting for fetch
  if (error) return <div>{error}</div> // Show error state if fetching fails
  return (
    <div>
      <h2>Route: {routeData?.route_name}</h2>
      {/* Render route details here */}
      {routeData && <pre>{JSON.stringify(routeData, null, 2)}</pre>} {/* Render route data as JSON for debugging */}
    </div>
  )
}

export default RouteLayout
