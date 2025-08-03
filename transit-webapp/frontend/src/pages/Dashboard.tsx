// src/pages/Dashboard.tsx
import React, { useContext, useEffect, useState, useMemo } from 'react'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { useAuth } from '../contexts/AuthContext'

const Dashboard: React.FC = () => {
  const { user, session } = useAuth();
  const globalData = useContext(GlobalDataContext)

  // Move all hooks to the top, before any conditional returns
  const metrics = useMemo(() => {
    if (!globalData) {
      return {
        totals: {
          totalStops: 0,
          totalRoutes: 0,
          totalLabels: 0,
          totalViolations: 0,
          totalPerformanceMetrics: 0,
          totalTravelSegments: 0,
          totalSamples: 0
        },
        violationsBySeverity: {},
        labelsByType: {},
        overallPunctuality: null,
        problematicRoutes: []
      }
    }

    const { stops, routes, labels, violations, time_types, travel_times, performance } = globalData

    const totalStops = Object.keys(stops).length
    const totalRoutes = Object.keys(routes).length
    
    // Convert labels/violations arrays to objects if needed
    const labelsObj = Array.isArray(labels) 
      ? Object.fromEntries(labels.map((label: any) => [label.entity_key, label]))
      : labels || {}
    const violationsObj = Array.isArray(violations)
      ? Object.fromEntries(violations.map((violation: any) => [violation.entity_key, violation]))
      : violations || {}

    const totalLabels = Object.keys(labelsObj).length
    const totalViolations = Object.keys(violationsObj).length
    const totalPerformanceMetrics = Object.keys(performance || {}).length
    const totalTravelSegments = travel_times?.length || 0

    // Analyze violations by severity
    const violationsBySeverity = Object.values(violationsObj).reduce((acc: Record<string, number>, violation: any) => {
      const severity = violation.severity || 1
      acc[severity] = (acc[severity] || 0) + 1
      return acc
    }, {})

    // Analyze labels by type
    const labelsByType = Object.values(labelsObj).reduce((acc: Record<string, number>, label: any) => {
      const type = label.label_type || 'unknown'
      acc[type] = (acc[type] || 0) + 1
      return acc
    }, {})

    // Calculate overall performance metrics
    let totalSamples = 0
    let totalOnTime = 0
    let totalTooEarly = 0
    let totalTooLate = 0
    let totalDelaySum = 0

    Object.values(performance || {}).forEach((perf: any) => {
      if (perf?.analytics?.punctuality) {
        const punct = perf.analytics.punctuality
        const samples = punct.sample_size || 0
        totalSamples += samples
        
        if (punct.punctuality_distribution?.percentages) {
          const pct = punct.punctuality_distribution.percentages
          totalOnTime += (pct.on_time || 0) * samples / 100
          totalTooEarly += (pct.too_early || 0) * samples / 100
          totalTooLate += (pct.too_late || 0) * samples / 100
        }

        if (punct.basic_statistics?.mean_delay && samples > 0) {
          totalDelaySum += punct.basic_statistics.mean_delay * samples
        }
      }
    })

    const overallPunctuality = totalSamples > 0 ? {
      onTime: (totalOnTime / totalSamples * 100).toFixed(1),
      tooEarly: (totalTooEarly / totalSamples * 100).toFixed(1),
      tooLate: (totalTooLate / totalSamples * 100).toFixed(1),
      avgDelay: (totalDelaySum / totalSamples).toFixed(1)
    } : null

    // Find routes with most issues
    const routeIssues = Object.values(violationsObj).reduce((acc: Record<string, number>, violation: any) => {
      const routeId = violation.route_id
      if (routeId) {
        acc[routeId] = (acc[routeId] || 0) + 1
      }
      return acc
    }, {})

    const problematicRoutes = Object.entries(routeIssues)
      .sort(([,a], [,b]) => (b as number) - (a as number))
      .slice(0, 5)
      .map(([routeId, count]) => ({
        routeId,
        routeName: routes[routeId]?.route_short_name || routeId,
        issueCount: count as number
      }))

    return {
      totals: {
        totalStops,
        totalRoutes,
        totalLabels,
        totalViolations,
        totalPerformanceMetrics,
        totalTravelSegments,
        totalSamples
      },
      violationsBySeverity,
      labelsByType,
      overallPunctuality,
      problematicRoutes
    }
  }, [globalData]) // Simplified dependency

  // Early returns after all hooks
  if (!session || !session.access_token || !session.refresh_token || !user) {
    throw new Error('No valid session or tokens available');
  }

  if (!globalData) {
    return <div className="p-6">Loading dashboard…</div>
  }

  const { time_types } = globalData

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Transit System Dashboard</h1>
        <div className="text-sm text-gray-500">
          {time_types?.length || 0} time periods tracked
        </div>
      </div>

      {/* Main Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card title="Routes" value={metrics.totals.totalRoutes}  />
        <Card title="Stops" value={metrics.totals.totalStops} />
        <Card title="Performance Metrics" value={metrics.totals.totalPerformanceMetrics}  />
        <Card title="Travel Segments" value={metrics.totals.totalTravelSegments}  />
      </div>

      {/* System Health Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Overall Punctuality */}
        {metrics.overallPunctuality && (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4"> System-Wide Punctuality</h3>
            <div className="text-sm text-gray-600 mb-4">
              Based on {metrics.totals.totalSamples.toLocaleString()} samples
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-green-600">On Time</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${metrics.overallPunctuality.onTime}%` }}
                    ></div>
                  </div>
                  <span className="font-medium w-12">{metrics.overallPunctuality.onTime}%</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-yellow-600">Too Early</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${metrics.overallPunctuality.tooEarly}%` }}
                    ></div>
                  </div>
                  <span className="font-medium w-12">{metrics.overallPunctuality.tooEarly}%</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-red-600">Too Late</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${metrics.overallPunctuality.tooLate}%` }}
                    ></div>
                  </div>
                  <span className="font-medium w-12">{metrics.overallPunctuality.tooLate}%</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* System Issues Summary */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4"> System Issues</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{metrics.totals.totalViolations}</div>
              <div className="text-sm text-gray-600">Total Violations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{metrics.totals.totalLabels}</div>
              <div className="text-sm text-gray-600">Total Labels</div>
            </div>
          </div>
          
          {/* Violations by Severity */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Violations by Severity</h4>
            <div className="space-y-1">
              {Object.entries(metrics.violationsBySeverity)
                .sort(([a], [b]) => Number(b) - Number(a))
                .map(([severity, count]) => (
                  <div key={severity} className="flex items-center justify-between text-sm">
                    <span className={`
                      ${Number(severity) >= 5 ? 'text-red-600' : 
                        Number(severity) >= 3 ? 'text-orange-600' : 'text-yellow-600'}
                    `}>
                      Severity {severity}
                    </span>
                    <span className="font-medium">{String(count)}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>

      {/* Routes with Most Issues */}
      {metrics.problematicRoutes.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4"> Routes Requiring Attention</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {metrics.problematicRoutes.map((route, index) => (
              <div key={route.routeId} className="text-center p-3 bg-red-50 rounded-lg">
                <div className="text-lg font-bold text-red-600">Route {route.routeName}</div>
                <div className="text-sm text-gray-600">{route.issueCount} issues</div>
                <div className="text-xs text-gray-500">#{index + 1} most issues</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Routes with Most Issues */}
      {/* {metrics.problematicStops.length > 0 &&*/ (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4"> Stops Requiring Attention</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {metrics.problematicRoutes.map((route, index) => (
              <div key={route.routeId} className="text-center p-3 bg-red-50 rounded-lg">
                <div className="text-lg font-bold text-red-600">Route {route.routeName}</div>
                <div className="text-sm text-gray-600">{route.issueCount} issues</div>
                <div className="text-xs text-gray-500">#{index + 1} most issues</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Label Types Distribution */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4"> System Labels Distribution</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {Object.entries(metrics.labelsByType)
            .sort(([,a], [,b]) => (b as number) - (a as number))
            .slice(0, 8)
            .map(([type, count]) => (
              <div key={type} className="text-center p-3 bg-blue-50 rounded-lg">
                <div className="text-lg font-bold text-blue-600">{String(count)}</div>
                <div className="text-xs text-gray-600 capitalize">
                  {type.replace(/_/g, ' ')}
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

interface CardProps {
  title: string
  value: number
  icon?: string
}

const Card: React.FC<CardProps> = ({ title, value, icon }) => (
  <div className="bg-white rounded-lg shadow p-6 flex flex-col items-center">
    {icon && <div className="text-2xl mb-2">{icon}</div>}
    <h2 className="text-lg font-medium text-gray-700">{title}</h2>
    <p className="text-3xl font-bold mt-2 text-gray-900">{value.toLocaleString()}</p>
  </div>
)

export default Dashboard