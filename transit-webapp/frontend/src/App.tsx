// src/App.tsx
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import RoutesList from './pages/RoutesList'
import StopsList  from './pages/StopsList'
import RouteLayout from './pages/route'
import StopLayout  from './pages/stop'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"               element={<Dashboard />} />
        <Route path="routes"          element={<RoutesList />} />
        <Route path="routes/:rid/*"   element={<RouteLayout />} />
        <Route path="stops"           element={<StopsList />} />
        <Route path="stops/:sid/*"    element={<StopLayout />} />
        <Route path="*"               element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
