import React, { useContext, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { useRouteData } from '../contexts/RouteDataContext'
import { useStopData } from '../contexts/StopDataContext'
import { PageHeader } from '../components/shared'
import { AccessControl } from '../components/shared'

interface DataListProps {
  type: 'routes' | 'stops'; // Type of data (either 'routes' or 'stops') // Only globalData needed
}

const DataList: React.FC<DataListProps> = ({ type}) => {
  
  const globalData = useContext(GlobalDataContext)
  const { setRouteId } = useRouteData()
  const { setParentId} = useStopData()
  const navigate = useNavigate()
  const [searchTerm, setSearchTerm] = useState('')

  // Get data directly from globalData
  const data = type === 'routes' ? globalData?.routes : globalData?.stops;
  
  // Filter data based on search term - must be called before any early returns
  const filteredData = useMemo(() => {
    if (!data || !searchTerm.trim()) {
      return data;
    }

    const searchLower = searchTerm.toLowerCase().trim();
    
    return Object.fromEntries(
      Object.entries(data).filter(([id, itemData]: [string, any]) => {
        if (type === 'routes') {
          // Search in route short name, long name, and route ID
          return (
            itemData.route_short_name?.toLowerCase().includes(searchLower) ||
            itemData.route_long_name?.toLowerCase().includes(searchLower) ||
            id.toLowerCase().includes(searchLower)
          );
        } else {
          // Search in stop name and parent station ID
          return (
            itemData.stop_name?.toLowerCase().includes(searchLower) ||
            id.toLowerCase().includes(searchLower)
          );
        }
      })
    );
  }, [data, searchTerm, type]);

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



  return (
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
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <input
          type="text"
          placeholder={type === 'routes' 
            ? "Search routes by name or ID..." 
            : "Search stops by name or station ID..."
          }
          className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        {searchTerm && (
          <button
            onClick={() => setSearchTerm('')}
            className="absolute inset-y-0 right-0 pr-3 flex items-center"
          >
            <svg className="h-5 w-5 text-gray-400 hover:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
      {/* Protected Content - System Health Section */}
      <AccessControl requireAdmin={true}>
      {/* Main Content */}
      <div className="space-y-4">
        {Object.entries(filteredData || {}).map(([id, itemData]: [string, any]) => {
          return (
            <div
              key={id}
              className="bg-white shadow-sm rounded-lg border border-gray-200 hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => handleItemClick(id)}
            >
              <div className="px-6 py-4">
                <div className="flex items-center justify-between">
                  {/* Icon and Main Info */}
                  <div className="flex items-center space-x-4 flex-1">
                    {/* Icon */}
                    <div className="flex-shrink-0">
                      {type === 'routes' ? (
                        <div className="w-10 h-10 bg-sky-100 rounded-lg flex items-center justify-center">
                          <span className="text-sm font-bold text-sky-600">
                            {itemData.route_short_name}
                          </span>
                        </div>
                      ) : (
                        <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                          <svg className="w-5 h-5 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                        </div>
                      )}
                    </div>

                    {/* Main Info */}
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {type === 'routes' 
                          ? `Route ${itemData.route_short_name}` 
                          : itemData.stop_name
                        }
                      </h3>
                      
                      {type === 'routes' ? (
                        <div className="text-sm text-gray-600 mt-1">
                          {itemData.route_long_name}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-600 mt-1">
                          Parent Station ID: {id}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Arrow indicator */}
                  <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                  </svg>
                </div>

                {/* Additional Details */}
                <div className="mt-3 pt-3 border-t border-gray-100">
                  {type === 'routes' && (
                    <div className="text-sm text-gray-600">
                      <span className="font-medium">Route ID:</span> {id}
                    </div>
                  )}

                  {/* Stop IDs for stops */}
                  {type === 'stops' && Array.isArray(itemData.stop_ids) && itemData.stop_ids.length > 1 && (
                    <div className="mt-3">
                      <div className="font-medium text-sm text-gray-700 mb-2">Associated Stop IDs:</div>
                      <div className="flex flex-wrap gap-1">
                        {itemData.stop_ids.slice(0, 5).map((stopId: string, index: number) => (
                          <span 
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-700"
                          >
                            {stopId}
                          </span>
                        ))}
                        {itemData.stop_ids.length > 5 && (
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-500">
                            +{itemData.stop_ids.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
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
      </AccessControl>
    </div>
  );
};

export default DataList;