// src/contexts/GlobalDataContext.tsx
import React, {
  createContext,
  ReactNode,
  useState,
  useEffect,
} from 'react'

// Define the shape of a single label entry
export interface LabelEntry {
  label_type: string
  description: string
  entity_key: string
  // other fields as needed…
}

// Define the shape of a single violation entry
export interface ViolationEntry {
  violation_type: string
  severity: string
  description: string
  entity_key: string
  // other fields as needed…
}

// Aggregate all global data into one interface
export interface GlobalData {
  labels: Record<string, LabelEntry>
  violations: Record<string, ViolationEntry>
  time_types: string[]
  stops: Record<string, any>
}

// Create the context
export const GlobalDataContext = createContext<GlobalData | null>(null)

// Provider props including children
interface ProviderProps {
  children: ReactNode
}

// The provider component
export const GlobalDataProvider: React.FC<ProviderProps> = ({ children }) => {
  const [data, setData] = useState<GlobalData | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/global/labels').then((r) => r.json()),
      fetch('/api/global/violations').then((r) => r.json()),
      fetch('/api/global/time_types').then((r) => r.json()),
      fetch('/api/global/stops').then((r) => r.json()),
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
  }, [])

  if (!data) return <div>Loading globals…</div>

  return (
    <GlobalDataContext.Provider value={data}>
      {children}
    </GlobalDataContext.Provider>
  )
}
