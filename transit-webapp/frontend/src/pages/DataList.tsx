import React from 'react';


interface DataListProps {
  data: any; // The data to be displayed (either routes or stops)
  type: 'routes' | 'stops'; // Type of data (either 'routes' or 'stops')
  setContextId: (id: string) => void; // Function to set the context ID (routeId or stopId)
}

const DataList: React.FC<DataListProps> = ({ data, type, setContextId }) => {
  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">{type === 'routes' ? 'All Routes' : 'All Stops'}</h2>
      <div className="space-y-4">
        {Object.entries(data).map(([id, meta]: [string, any]) => (
          <div
            key={id}
            className="block bg-white rounded-2xl shadow-sm hover:shadow-md transition p-5 cursor-pointer"
            onClick={() => {
              setContextId(id);
            }}
          >
            <div>
              <h3 className="text-lg font-medium">{meta.route_short_name || meta.stop_name}</h3>
              <div className="text-sm text-gray-500 mt-2 space-y-2">
                {type === 'routes' ? (
                  <>
                    <div>
                      <span className="font-medium">Long Name:</span> {meta.route_long_name}
                    </div>
                    <div>
                      <span className="font-medium">Route ID:</span> {id}
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <span className="font-medium">Parent Station:</span> {meta.parent_station}
                    </div>
                    <div>
                      <div className="font-medium">Stop IDs:</div>
                      {/* Type the map function explicitly */}
                      {Array.isArray(meta.stop_ids) && (
                        <ul className="list-disc list-inside mt-1 space-y-1">
                          {meta.stop_ids.map((stopId: string, index: number) => (
                            <li key={index}>{stopId}</li>  
                          ))}
                        </ul>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DataList;
