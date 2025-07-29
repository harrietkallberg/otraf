import React, {
  createContext,
  ReactNode,
  useState,
  useEffect,
} from 'react'
import { useAuth } from './AuthContext'

export interface LabelEntry {
  label_type: string
  description: string
  entity_key: string
}

export interface ViolationEntry {
  violation_type: string
  severity: string
  description: string
  entity_key: string
}

export interface GlobalData {
  labels: Record<string, LabelEntry>
  violations: Record<string, ViolationEntry>
  time_types: string[]
  stops: Record<string, any>
}

export const GlobalDataContext = createContext<GlobalData | null>(null)

interface ProviderProps {
  children: ReactNode
}

export const GlobalDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [data, setData] = useState<GlobalData | null>(null)
  const { user } = useAuth()

  useEffect(() => {
    if (!user) return

    const headers = { 'X-User-Id': user.id }

    Promise.all([
      fetch('/api/global/labels', { headers }).then(r => r.json()),
      fetch('/api/global/violations', { headers }).then(r => r.json()),
      fetch('/api/global/time_types', { headers }).then(r => r.json()),
      fetch('/api/global/stops', { headers }).then(r => r.json()),
    ])
      .then(([labels, violations, time_types, stops]) =>
        setData({
          labels: labels as Record<string, LabelEntry>,
          violations: violations as Record<string, ViolationEntry>,
          time_types,
          stops: stops as Record<string, any>,
        })
      )
      .catch(console.error)
  }, [user])

  if (!data) return <div>Loading globals…</div>

  return (
    <GlobalDataContext.Provider value={data}>
      {children}
    </GlobalDataContext.Provider>
  )
}
