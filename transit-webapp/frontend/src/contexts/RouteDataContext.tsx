import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react'
import { useAuth } from './AuthContext'
import * as DataInterfaces from './DataInterfaces'

export interface RouteDataContextType {
  routeId: string | null;
  setRouteId: (id: string | null) => void;
  routeData: DataInterfaces.RouteData | null;  // Correctly using RouteData interface
  setRouteData: (data: any) => void;
  isLoading: boolean;
  error: string | null;
}

export const RouteDataContext = createContext<RouteDataContextType | null>(null)

interface ProviderProps {
  children: ReactNode
}

export const RouteDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [routeId, setRouteId] = useState<string | null>(null)
  const [routeData, setRouteData] = useState<DataInterfaces.RouteData | null>(null)  // Correct type for routeData
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const { user, session, isLoading: authLoading } = useAuth()

  useEffect(() => {
    if (!routeId || !user || !session || authLoading) {
      if (!routeId) {
        setRouteData(null)
        setIsLoading(false)
        setError(null)
      }
      return
    }

    // If we already have data for this routeId, don't fetch again
    if (routeData && routeData.route_id === routeId) {
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    setError(null)
    setRouteData(null)  // Clear old data immediately when starting new fetch

    const headers = {
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,
    }

    console.log('RouteDataContext fetching for routeId:', routeId)

    fetch(`/api/routes/${routeId}`, { headers })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`)
        }
        return res.json()
      })
      .then((data) => {
        console.log('RouteDataContext received data for routeId:', routeId)
        setRouteData(data)  // Data is typed as RouteData
        setError(null)
      })
      .catch((err) => {
        console.error('RouteDataContext error:', err)
        setError('Failed to load route data.')
        setRouteData(null)
      })
      .finally(() => {
        setIsLoading(false)
      })
  }, [routeId, user, session, authLoading])

  return (
    <RouteDataContext.Provider value={{ 
      routeId, 
      setRouteId, 
      routeData, 
      setRouteData, 
      isLoading, 
      error 
    }}>
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
