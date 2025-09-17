import React, { useContext, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { useRouteData } from '../contexts/RouteDataContext'
import { useStopData } from '../contexts/StopDataContext'
import { PageHeader } from '../components/shared'
import { AccessControl } from '../components/shared'

interface DataListProps {
  type: 'routes' | 'stops'; // Type of data (either 'routes' or 'stops')
}

const DataList: React.FC<DataListProps> = ({ type }) => {
  const globalData = useContext(GlobalDataContext)
  const { setRouteId } = useRouteData()
  const { setParentId } = useStopData()
  const navigate = useNavigate()
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState<'violations' | 'labels' | 'on_time' | 'none'>('none')

  // Get data based on type ('routes' or 'stops')
  const data = type === 'routes' ? globalData?.routes : globalData?.stops;

  // Function to compare values for sorting based on chosen field
  const sortData = (a: any, b: any) => {
    switch (sortBy) {
      case 'violations':
        return (a.violation_count || 0) - (b.violation_count || 0);
      case 'labels':
        return (a.label_count || 0) - (b.label_count || 0);
      case 'on_time':
        return (b.on_time_pct || 0) - (a.on_time_pct || 0); // Sorting descending by on-time percentage
      default:
        return 0; // No sorting
    }
  }

  // Filter data based on search term
  const filteredData = useMemo(() => {
    if (!data || !searchTerm.trim()) {
      return data;
    }

    const searchLower = searchTerm.toLowerCase().trim();
    return Object.fromEntries(
      Object.entries(data).filter(([id, itemData]: [string, any]) => {
        if (type === 'routes') {
          return (
            itemData.route_short_name?.toLowerCase().includes(searchLower) ||
            itemData.route_long_name?.toLowerCase().includes(searchLower) ||
            id.toLowerCase().includes(searchLower)
          );
        } else {
          return (
            itemData.stop_name?.toLowerCase().includes(searchLower) ||
            id.toLowerCase().includes(searchLower)
          );
        }
      })
    );
  }, [data, searchTerm, type]);

  // Handle sort change
  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSortBy(e.target.value as 'violations' | 'labels' | 'on_time' | 'none');
  }

  // Helper function to handle click and navigation
  const handleItemClick = (id: string) => {
    if (type === 'routes') {
      setRouteId(id);
      navigate(`/routes/${id}`); // Navigate to route detail page
    } else {
      setParentId(id);
      navigate(`/stops/${id}`); // Navigate to stop detail page
    }
  };

  if (!globalData) {
    return <div className="px-6 py-4">Loading routes...</div>
  }

  if (!data) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold mb-6">
          {type === 'routes' ? 'All Routes' : 'All Stops'}
        </h1>
        <div className="text-gray-500">No {type} available</div>
      </div>
    );
  }

  const helpText = type === 'routes' 
    ? "This page displays all available routes within the system. You can search for a specific route using the search bar, or browse through the list. To view detailed information about a route, click on its respective tile."
    : "This page shows all the stops available in the system. You can search for a specific stop using the search bar or explore through the list. To view detailed information about a stop, click on its respective tile.";

  return (
    <AccessControl>
      <div className="p-6 space-y-6">
        <PageHeader 
          title={type === 'routes' ? 'All Routes' : 'All Stops'}
          helpText={helpText}
          subtitle={type === 'routes' 
            ? `${Object.keys(data || {}).length} routes available`
            : `${Object.keys(data || {}).length} stops available`
          }
        />

        {/* Search Bar */}
        <div className="relative mb-6">
          <input
            type="text"
            placeholder={type === 'routes' ? "Search routes..." : "Search stops..."}
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {/* Sort Dropdown */}
        <div className="mb-4">
          <label className="block text-sm text-gray-700">Sort by:</label>
          <select 
            value={sortBy}
            onChange={handleSortChange}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="none">None</option>
            <option value="violations">Violations</option>
            <option value="labels">Labels</option>
            <option value="on_time">On-time performance</option>
          </select>
        </div>

        {/* List Data */}
        <div className="space-y-4">
          {Object.entries(filteredData || {})
            .sort(([idA, dataA], [idB, dataB]) => sortData(dataA, dataB))
            .map(([id, itemData]: [string, any]) => {
              // Extract necessary data from route/stop navigation
              const violations = itemData.route_summary?.route_summary?.stop_topology?.violations_by_type || itemData.stop_summary?.stop_summary?.direction_topology?.violations_by_type || 0;
              const labels = itemData.route_summary?.route_summary?.stop_topology?.labels_by_type || itemData.stop_summary?.stop_summary?.direction_topology?.labels_by_type || 0;
              const onTimePerformance = itemData.route_summary?.performance_summary?.overall_on_time_rate || itemData.stop_summary?.performance_summary?.overall_on_time_rate || 0;
              
              return (
                <div
                  key={id}
                  className="bg-white shadow-sm rounded-lg border border-gray-200 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => handleItemClick(id)}
                >
                  <div className="px-6 py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 flex-1">
                        <div className="flex-shrink-0">
                          {/* Display Icon or Image for Routes/Stops */}
                          {type === 'routes' ? (
                            <div className="w-10 h-10 bg-sky-100 rounded-lg flex items-center justify-center">
                              <span className="text-sm font-bold text-sky-600">
                                {itemData.route_short_name}
                              </span>
                            </div>
                          ) : (
                            <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                              <span className="text-sm">{itemData.stop_name}</span>
                            </div>
                          )}
                        </div>

                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-gray-900">
                            {type === 'routes' ? `Route ${itemData.route_short_name}` : itemData.stop_name}
                          </h3>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600">
                      <div><strong>Violations:</strong> {violations}</div>
                      <div><strong>Labels:</strong> {labels}</div>
                      <div><strong>On-time Performance:</strong> {onTimePerformance}%</div>
                    </div>
                  </div>
                </div>
              );
            })
          }
        </div>

        {/* Empty State */}
        {Object.keys(filteredData || {}).length === 0 && Object.keys(data).length > 0 && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-lg mb-2">
              No {type} found matching "{searchTerm}"
            </div>
            <button
              onClick={() => setSearchTerm('')}
              className="text-blue-600 hover:text-blue-800 text-sm"
            >
              Clear search
            </button>
          </div>
        )}

        {/* No Data State */}
        {Object.keys(data).length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-lg">
              No {type} available
            </div>
          </div>
        )}
      </div>
    </AccessControl>
  );
};

export default DataList;
