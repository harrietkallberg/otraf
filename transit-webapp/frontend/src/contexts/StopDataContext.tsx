import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react'
import { useAuth } from './AuthContext'
import * as DataInterfaces from '../shared/types'

// Update the stopData type with stop_summary
export interface StopDataContextType {
  parentId: string | null;
  setParentId: (id: string | null) => void;
  stopData: DataInterfaces.StopData | null;  // Correctly using StopData interface
  setStopData: (data: any) => void;
  isLoading: boolean;
  error: string | null;
}

export const StopDataContext = createContext<StopDataContextType | null>(null)

interface ProviderProps {
  children: ReactNode
}

export const StopDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [parentId, setParentId] = useState<string | null>(null)
  const [stopData, setStopData] = useState<DataInterfaces.StopData | null>(null)  // Correct type for stopData
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const { user, session, isLoading: authLoading } = useAuth()

  useEffect(() => {
    if (!parentId || !user || !session || authLoading) {
      if (!parentId) {
        setStopData(null)
        setIsLoading(false)
        setError(null)
      }
      return
    }

    // If we already have data for this parentId, don't fetch again
    if (stopData && stopData.parent_station === parentId) {
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    setError(null)
    setStopData(null)  // Clear old data immediately when starting new fetch

    const headers = {
      'X-User-Id': user.id,
      'Authorization': `Bearer ${session.access_token}`,
      'X-Refresh-Token': session.refresh_token,
    }

    console.log('StopDataContext fetching for parentId:', parentId)

    fetch(`/api/stops/${parentId}`, { headers })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`)
        }
        return res.json()
      })
      .then((data) => {
        console.log('StopDataContext received data for parentId:', parentId)
        setStopData(data)  // Data is typed as StopData
        setError(null)
      })
      .catch((err) => {
        console.error('StopDataContext error:', err)
        setError('Failed to load stop data.')
        setStopData(null)
      })
      .finally(() => {
        setIsLoading(false)
      })
  }, [parentId, user, session, authLoading, stopData])

  return (
    <StopDataContext.Provider value={{ 
      parentId, 
      setParentId, 
      stopData, 
      setStopData, 
      isLoading, 
      error 
    }}>
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
