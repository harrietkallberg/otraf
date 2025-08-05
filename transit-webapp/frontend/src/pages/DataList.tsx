import React, { useContext } from 'react'
import { useNavigate } from 'react-router-dom'
import { GlobalDataContext } from '../contexts/GlobalDataContext'
import { useRouteData } from '../contexts/RouteDataContext'
import { useStopData } from '../contexts/StopDataContext'

interface DataListProps {
  type: 'routes' | 'stops'; // Type of data (either 'routes' or 'stops') // Only globalData needed
}

const DataList: React.FC<DataListProps> = ({ type}) => {
  
  const globalData = useContext(GlobalDataContext)
  const { setRouteId } = useRouteData()
  const { setParentId} = useStopData()
  const navigate = useNavigate()

  if (!globalData) {
    return <div className="px-6 py-4">Loading routes...</div>
  }

  // Get data directly from globalData
  const data = type === 'routes' ? globalData.routes : globalData.stops;
  
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
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">
          {type === 'routes' ? 'All Routes' : 'All Stops'}
        </h1>
        <div className="text-sm text-gray-500">
          {Object.keys(data).length} {type} available
        </div>
      </div>

      {/* Main Content */}
      <div className="space-y-4">
        {Object.entries(data).map(([id, itemData]: [string, any]) => {
          return (
            <div
              key={id}
              className="bg-white shadow-sm rounded-lg border border-gray-200 hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => handleItemClick(id)}
            >
              <div className="px-6 py-4">
                <div className="flex items-center justify-between">
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
      {Object.keys(data).length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg">
            No {type} available
          </div>
        </div>
      )}
    </div>
  );
};

export default DataList;