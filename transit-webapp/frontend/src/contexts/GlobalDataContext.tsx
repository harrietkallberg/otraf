import React, { createContext, ReactNode, useState, useEffect } from 'react'
import { useAuth } from './AuthContext'

export interface GlobalData {
  labels: Record<string, any>
  violations: Record<string, any>
  time_types: string[]
  stops: Record<string, any>
}

// Create a context that can be GlobalData or null
export const GlobalDataContext = createContext<GlobalData | null>(null)

interface ProviderProps {
  children: ReactNode
}

export const GlobalDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [data, setData] = useState<GlobalData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { user, session, isLoading } = useAuth()

  useEffect(() => {
    // Only fetch data if user is authenticated and we have a session
    if (!user || !session || isLoading) {
      console.log('User not ready:', { user: !!user, session: !!session, isLoading })
      setData(null)
      setError(null)
      return
    }

    // Send both user ID, session token, and refresh token
    const headers = { 
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,  // Pass refresh token in a custom header
    }

    console.log('Headers being sent:', {
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token.substring(0, 20)}...`, // Log partial token for debugging
      'X-Refresh-Token': `Bearer ${session.refresh_token.substring(0, 20)}...` // Log partial refresh token for debugging
    })

    // Fetch global data using both tokens and user ID in headers
    Promise.all([
      fetch('/api/global/labels', { headers }).then(r => r.json()),
      fetch('/api/global/violations', { headers }).then(r => r.json()),
      fetch('/api/global/time_types', { headers }).then(r => r.json()),
      fetch('/api/global/stops', { headers }).then(r => r.json()),
    ])
      .then(([labels, violations, time_types, stops]) =>
        setData({
          labels: labels,
          violations: violations,
          time_types,
          stops: stops,
        })
      )
      .catch((err) => {
        console.error(err)
        setError('Failed to load global data. Please try again later.')
      })
  }, [user, session, isLoading])

  return (
    <GlobalDataContext.Provider value={data}>
      {children}
    </GlobalDataContext.Provider>
  )
}

// Custom hook for easier usage with proper typing
export const useGlobalData = () => {
  const context = React.useContext(GlobalDataContext)
  return context // This can be null, components should handle this
}
