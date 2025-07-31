import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react'
import { useAuth } from './AuthContext'

export interface RouteDataContextType {
  routeId: string | null
  setRouteId: (id: string | null) => void
  routeData: any | null
  setRouteData: (data: any) => void
}

export const RouteDataContext = createContext<RouteDataContextType | null>(null)

interface ProviderProps {
  children: ReactNode
}

export const RouteDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [routeId, setRouteId] = useState<string | null>(null)
  const [routeData, setRouteData] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { user, session, isLoading } = useAuth()

  useEffect(() => {
    if (!routeId || !user || !session || isLoading) {
      return
    }

    const headers = {
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,  // Pass refresh token in a custom header
    }

    fetch(`/api/routes/${routeId}/navigation_structure`, { headers })
      .then((res) => res.json())
      .then((data) => setRouteData(data))
      .catch((err) => {
        console.error('Error fetching route data:', err)
        setError('Failed to load route data.')
      })
  }, [routeId, user, session, isLoading]) // Trigger fetch when routeId changes

  return (
    <RouteDataContext.Provider value={{ routeId, setRouteId, routeData, setRouteData }}>
      {children}
    </RouteDataContext.Provider>
  )
}

export const useRouteData = () => {
  const context = useContext(RouteDataContext)
  if (!context) {
    throw new Error('useRouteData must be used within a RouteDataProvider')
  }
  return context
}
