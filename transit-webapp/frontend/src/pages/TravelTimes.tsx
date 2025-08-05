// src/pages/TravelTimes.tsx
import React, { useEffect, useState, useContext } from 'react'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { ProgressiveSearchFilters, FilterConfig } from '../components/shared/ProgressiveSearchFilters'
import { useProgressiveSearch, FilterValidationRule } from '../hooks/useProgressiveSearch'

interface ByRouteEntry {
  route_id: number
  direction_id: string
  mean: number | null
  sample_size: number
}

interface TravelSegment {
  from_stop_id: string
  to_stop_id: string
  from_stop_name: string
  to_stop_name: string
  time_type: string
  aggregated: {
    mean: number | null
    sample_size: number
  }
  by_route: ByRouteEntry[]
}

const TravelTimes: React.FC = () => {
  const [segments, setSegments] = useState<TravelSegment[]>([])
  const [expandedSegments, setExpandedSegments] = useState<Set<number>>(new Set())
  const globalData = useContext(GlobalDataContext)

  // Set segments directly from travel_times when globalData is available
  useEffect(() => {
    if (!globalData?.travel_times) return
    console.log('Setting travel times data:', globalData.travel_times)
    setSegments(globalData.travel_times)
  }, [globalData])

  const hasLetters = (s: string) => /\D/.test(s)

  const toggleSegmentDetails = (index: number) => {
    const newExpanded = new Set(expandedSegments)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSegments(newExpanded)
  }

  // Define filter rules for progressive search
  const filterRules: FilterValidationRule<TravelSegment>[] = [
    {
      key: 'route',
      validateItem: (item, filters) => !filters.route || item.by_route?.some(r => r.route_id === Number(filters.route)),
      extractOptions: (item) => item.by_route?.map(r => r.route_id) || [],
    },
    {
      key: 'fromName',
      validateItem: (item, filters) => !filters.fromName || item.from_stop_name === filters.fromName,
      extractOptions: (item) => hasLetters(item.from_stop_name) ? item.from_stop_name : [],
      dependencies: ['toName']
    },
    {
      key: 'toName',
      validateItem: (item, filters) => !filters.toName || item.to_stop_name === filters.toName,
      extractOptions: (item) => hasLetters(item.to_stop_name) ? item.to_stop_name : [],
    },
    {
      key: 'timeType',
      validateItem: (item, filters) => !filters.timeType || item.time_type === filters.timeType,
      extractOptions: (item) => item.time_type,
    }
  ];

  // Use the progressive search hook
  const {
    filters,
    setFilter,
    filteredData: filteredSegments,
    availableOptions
  } = useProgressiveSearch({
    data: segments,
    initialFilters: {},
    filterRules,
    hasValidData: (item) => {
      return (typeof item.aggregated?.mean === 'number') ||
             (item.by_route?.some(r => typeof r.mean === 'number'));
    }
  });

  // Create filter configurations
  const filterConfigs: FilterConfig[] = [
    {
      key: 'route',
      label: 'Route',
      placeholder: '(all routes)',
      value: filters.route || '',
      onChange: (value) => setFilter('route', value),
      options: Array.from(availableOptions.route || [])
        .sort((a, b) => Number(a) - Number(b))
        .map(routeId => ({
          value: routeId,
          label: `Route ${globalData?.routes?.[Number(routeId)]?.route_short_name || routeId}`
        }))
    },
    {
      key: 'fromName',
      label: 'From Stop',
      placeholder: '(any stop)',
      value: filters.fromName || '',
      onChange: (value) => setFilter('fromName', value),
      options: Array.from(availableOptions.fromName || [])
        .sort((a, b) => a.localeCompare(b))
        .map(name => ({
          value: name,
          label: name
        }))
    },
    {
      key: 'toName',
      label: 'To Stop',
      placeholder: '(any stop)',
      value: filters.toName || '',
      onChange: (value) => setFilter('toName', value),
      options: Array.from(availableOptions.toName || [])
        .sort((a, b) => a.localeCompare(b))
        .map(name => ({
          value: name,
          label: name
        }))
    },
    {
      key: 'timeType',
      label: 'Time Type',
      placeholder: '(any time)',
      value: filters.timeType || '',
      onChange: (value) => setFilter('timeType', value),
      options: Array.from(availableOptions.timeType || [])
        .sort()
        .map(timeType => ({
          value: timeType,
          label: timeType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
        }))
    }
  ];

  if (!globalData) {
    return <div className="p-6">Loading travel times...</div>
  }

  // Add additional safety check
  if (!Array.isArray(globalData.travel_times)) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <h3 className="font-medium">Error Loading Travel Times</h3>
          <p className="text-sm mt-1">Travel times data is not in the expected format.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Travel Times</h1>
        <div className="text-sm text-gray-500">
          {filteredSegments.length} of {segments.length} segments
        </div>
      </div>

      {/* Progressive Search Filters */}
      <ProgressiveSearchFilters
        title="Filter Travel Times"
        subtitle={`Showing ${filteredSegments.length} of ${segments.length} travel segments`}
        filters={filterConfigs}
        className="mb-6"
      />

      {/* Travel Segments */}
      <div className="space-y-4">
        {filteredSegments.map((s, idx) => (
          <div key={idx} className="bg-white shadow-sm rounded-lg border border-gray-200">
            <div className="px-6 py-4">
              <div className="flex items-center justify-between">
                {/* Icon and Main Info */}
                <div className="flex items-center space-x-4 flex-1">
                  {/* Travel Time Icon */}
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                      <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                        <circle cx="4" cy="12" r="3" fill="none" />
                        <circle cx="20" cy="12" r="3" fill="none" />
                        <line x1="7" y1="12" x2="17" y2="12" strokeLinecap="round" />
                      </svg>
                    </div>
                  </div>

                 {/* Segment Info */}
                <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900">
                        {s.from_stop_name} → {s.to_stop_name}
                    </h3>
                    <div className="text-sm text-gray-600 mt-1">
                        {s.time_type?.replace('_', ' ').toUpperCase() || 'UNKNOWN'} {' '}
                        {!filters.route && typeof s.aggregated?.mean === 'number' ? (
                        // Show the dot and the mean if available
                        `• ${s.aggregated?.mean.toFixed(1)}s average across routes`
                        ) : (
                        // Show just the dot if no data is available
                        ' '
                        )}
                        {s.aggregated?.sample_size > 0 && (
                        <span className="text-gray-500">
                            {' '}• {s.aggregated.sample_size.toLocaleString()} samples
                        </span>
                        )}
                    </div>
                </div>
                </div>
                {/* Expand/Collapse Button */}
                {!filters.route && s.by_route && s.by_route.length > 0 && (
                  <button
                    onClick={() => toggleSegmentDetails(idx)}
                    className="flex items-center text-indigo-600 hover:text-indigo-800 text-sm"
                  >
                    {expandedSegments.has(idx) ? 'Hide Details' : 'Show Details'}
                    <svg className={`w-4 h-4 ml-1 transition-transform ${expandedSegments.has(idx) ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                )}
              </div>

              {/* Route-specific data or expanded details */}
              <div className="mt-3 pt-3 border-t border-gray-100">
                {filters.route ? (
                  // When a route is selected, show only that route's data
                  <div className="space-y-2">
                    {s.by_route
                      ?.filter(r => r.route_id === Number(filters.route))
                      .map((r, i) => (
                        <div key={i} className="flex items-center space-x-3">
                          <div className="w-6 h-6 bg-sky-100 rounded flex items-center justify-center">
                            <span className="text-xs font-bold text-sky-600">
                              {globalData.routes?.[r.route_id]?.route_short_name || r.route_id}
                            </span>
                          </div>
                          <div className="flex-1">
                            <span className="text-sm text-gray-700">
                              {typeof r.mean === 'number'
                                ? <span className="font-semibold">{r.mean.toFixed(1)}s</span>
                                : <span className="text-gray-500">n/a</span>}
                              {r.sample_size > 0 && (
                                <span className="text-gray-500 ml-2">
                                  ({r.sample_size.toLocaleString()} samples, Direction {r.direction_id}:{' '} )
                                </span>
                              )}
                            </span>
                          </div>
                        </div>
                      ))
                    }
                  </div>
                ) : (
                  // When no route is selected, show expandable details
                  expandedSegments.has(idx) && s.by_route && s.by_route.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">By Route & Direction:</h4>
                      {s.by_route.map((r, i) => (
                        <div key={i} className="flex items-center space-x-3">
                          <div className="w-6 h-6 bg-sky-100 rounded flex items-center justify-center">
                            <span className="text-xs font-bold text-sky-600">
                              {globalData.routes?.[r.route_id]?.route_short_name || r.route_id}
                            </span>
                          </div>
                          <div className="flex-1">
                            <span className="text-sm text-gray-700">
                              Direction {r.direction_id}:{' '}
                              {typeof r.mean === 'number'
                                ? <span className="font-semibold">{r.mean.toFixed(1)}s</span>
                                : <span className="text-gray-500">n/a</span>}
                              {r.sample_size > 0 && (
                                <span className="text-gray-500 ml-2">
                                  ({r.sample_size.toLocaleString()} samples)
                                </span>
                              )}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Empty State */}
        {filteredSegments.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-lg mb-2">
              No matching travel segments found
            </div>
            <p className="text-gray-500 text-sm">
              Try adjusting your filters to see available travel time data.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default TravelTimes;