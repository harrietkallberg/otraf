import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'

import { AuthProvider } from './contexts/AuthContext'
import { GlobalDataProvider } from './contexts/GlobalDataContext'
import { RouteDataProvider } from './contexts/RouteDataContext'
import { StopDataProvider } from './contexts/StopDataContext'

import PrivateRoute from './components/PrivateRoute'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'

import Dashboard from './pages/Dashboard'
import RoutesList from './pages/RoutesList'
import StopsList from './pages/StopsList'
import TravelTimes from './pages/TravelTimes'
import ExportCsvPage from './pages/ExportCsv'
import ExplorePage from './pages/ExplorePage'
import UnifiedLayoutPage from './pages/UnifiedLayout'

export default function App() {
  return (
    <AuthProvider>
      <GlobalDataProvider>
        <RouteDataProvider>
          <StopDataProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />

              <Route
                element={
                  <PrivateRoute>
                    <Layout />
                  </PrivateRoute>
                }
              >
                <Route path="/" element={<Dashboard />} />
                <Route path="routes" element={<RoutesList />} />
                <Route path="routes/:routeId" element={<UnifiedLayoutPage />} />
                <Route path="stops" element={<StopsList />} />
                <Route path="stops/:parentId" element={<UnifiedLayoutPage />} />
                <Route path="travel-times" element={<TravelTimes />} />
                <Route path="export-csv" element={<ExportCsvPage />} />
                <Route path="explore-logs" element={<ExplorePage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </StopDataProvider>
        </RouteDataProvider>
      </GlobalDataProvider>
    </AuthProvider>
  )
}