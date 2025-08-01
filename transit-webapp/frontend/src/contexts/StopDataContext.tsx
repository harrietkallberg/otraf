import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react'
import { useAuth } from './AuthContext'

export interface StopDataContextType {
  parentId: string | null // Changed from stopId to parentId
  setParentId: (id: string | null) => void // Adjusted to use parentId
  stopData: any | null
  setStopData: (data: any) => void
}

export const StopDataContext = createContext<StopDataContextType | null>(null)

interface ProviderProps {
  children: ReactNode
}

export const StopDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [parentId, setParentId] = useState<string | null>(null) // Parent ID instead of stop ID/name
  const [stopData, setStopData] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { user, session, isLoading } = useAuth()

  useEffect(() => {
    if (!parentId || !user || !session?.access_token || isLoading) {
      return
    }

    const headers = {
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,
    }

    // Fetching data based on parentId
    fetch(`/api/stops/${parentId}`, { headers })  // We now pass parentId instead of stop name or stopId
      .then((res) => res.json())
      .then((data) => {
        if (data) {
          setStopData(data) // Aggregate stop data across all stop_ids for the stop name
        } else {
          setError('No data found for the selected stop')
        }
      })
      .catch((err) => {
        console.error(err)
        setError('Failed to load stop data')
      })
  }, [parentId, user, session, isLoading])

  return (
    <StopDataContext.Provider value={{ parentId, setParentId, stopData, setStopData }}>
      {children}
    </StopDataContext.Provider>
  )
}

export const useStopData = () => {
  const context = useContext(StopDataContext)
  if (!context) {
    throw new Error('useStopData must be used within a StopDataProvider')
  }
  return context
}
