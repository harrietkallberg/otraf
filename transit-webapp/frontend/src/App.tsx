// src/App.tsx
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { GlobalDataProvider } from './contexts/GlobalDataContext'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import RoutesList from './pages/RoutesList'
import StopsList  from './pages/StopsList'
import RouteLayout from './pages/route'
import StopLayout  from './pages/stop'
import TravelTimes from './pages/TravelTimes'
import ExportCsvPage from './pages/ExportCsv'

export default function App() {
  return (
    <GlobalDataProvider>
      <Layout>
        <Routes>
          <Route path="/"               element={<Dashboard />} />
          <Route path="routes"          element={<RoutesList />} />
          <Route path="routes/:rid/*"   element={<RouteLayout />} />
          <Route path="stops"           element={<StopsList />} />
          <Route path="stops/:sid/*"    element={<StopLayout />} />
          <Route path="*"               element={<Navigate to="/" replace />} />
          <Route path="travel-times" element={<TravelTimes />} />
          <Route path="/export-csv" element={<ExportCsvPage />} />
        </Routes>
      </Layout>
    </GlobalDataProvider>
  )
}

