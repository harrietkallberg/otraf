import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'

import { AuthProvider } from './contexts/AuthContext'
import { GlobalDataProvider } from './contexts/GlobalDataContext'
import { RouteDataProvider } from './contexts/RouteDataContext'  // Import RouteDataProvider
import { StopDataProvider } from './contexts/StopDataContext'  // Import StopDataProvider

import PrivateRoute from './components/PrivateRoute'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'

import Dashboard from './pages/Dashboard'
import RoutesList from './pages/RoutesList'
import StopsList from './pages/StopsList'
import RouteLayout from './pages/RouteLayout'
import StopLayout from './pages/StopLayout'
import TravelTimes from './pages/TravelTimes'
import ExportCsvPage from './pages/ExportCsv'

export default function App() {
  return (
    <AuthProvider>
      <GlobalDataProvider>
        <RouteDataProvider> {/* Wrap the entire routing section with RouteDataProvider */}
          <StopDataProvider> {/* Wrap Stop-related pages with StopDataProvider */}
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
                <Route path="routes" element={<RoutesList />} /> {/* Route list page */}

                {/* Now RouteLayout and RoutesList both have access to RouteDataContext */}
                <Route path="routes/:routeId" element={<RouteLayout />} /> {/* Route Layout */}

                {/* Stop-related routes are now wrapped with StopDataProvider */}
                <Route path="stops" element={<StopsList />} />
                <Route path="stops/:sid" element={<StopLayout />} />
                <Route path="travel-times" element={<TravelTimes />} />
                <Route path="export-csv" element={<ExportCsvPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </StopDataProvider>
        </RouteDataProvider>
      </GlobalDataProvider>
    </AuthProvider>
  )
}
