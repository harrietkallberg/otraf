import React from 'react';
import { DirectionData, StopIdData } from '../contexts/DataInterfaces';

interface StopSequenceProps {
  direction: DirectionData;
  routeId: string;
  directionId: string;
  globalData: any;
}

const StopSequence: React.FC<StopSequenceProps> = ({ 
  direction, 
  routeId, 
  directionId, 
  globalData 
}) => {
  const stopIdsInDirection = direction.stop_ids_in_direction;
  
  if (!stopIdsInDirection) {
    return <div className="text-gray-500 text-sm">No stops available for this direction.</div>;
  }

  // Sort stops by position
  const sortedStops = Object.entries(stopIdsInDirection)
    .sort(([a], [b]) => parseInt(a) - parseInt(b))
    .map(([position, stopData]: [string, any]) => ({
      position: parseInt(position),
      ...stopData
    }));

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h4 className="text-lg font-semibold mb-4 text-gray-900">Canonical Stop Sequence</h4>
      
      <div className="space-y-3">
        {sortedStops.map((stop) => (
          <StopIdCard 
            key={stop.stop_id}
            stopData={stop}
            routeId={routeId}
            directionId={directionId}
            globalData={globalData}
          />
        ))}
      </div>
    </div>
  );
};

interface StopIdCardProps {
  stopData: any; // StopIdData with position
  routeId: string;
  directionId: string;
  globalData: any;
}

const StopIdCard: React.FC<StopIdCardProps> = ({ 
  stopData, 
  routeId, 
  directionId, 
  globalData 
}) => {
  const isRegulatory = stopData.stop_id_label_keys?.some((key: string) => 
    key.includes('regulatory_stops')
  );

  const labelCount = stopData.stop_id_label_keys?.length || 0;
  const violationCount = stopData.stop_id_violation_keys?.length || 0;
  const performanceCount = stopData.stop_id_performance_keys?.length || 0;

  return (
    <div className={`border rounded-lg p-4 transition-all ${
      isRegulatory 
        ? 'border-amber-200 bg-amber-50' 
        : 'border-gray-200 bg-white hover:bg-gray-50'
    }`}>
      <div className="flex items-center justify-between">
        {/* Stop Info */}
        <div className="flex items-center space-x-4">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
            isRegulatory 
              ? 'bg-amber-200 text-amber-800' 
              : 'bg-blue-100 text-blue-800'
          }`}>
            {stopData.position}
          </div>
          
          <div>
            <h5 className="font-medium text-gray-900">{stopData.stop_name}</h5>
            <p className="text-sm text-gray-600">
              Stop ID: {stopData.stop_id} • Parent: {stopData.parent_station}
            </p>
          </div>
        </div>

        {/* Badges */}
        <div className="flex items-center space-x-2">
          {isRegulatory && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-amber-200 text-amber-800">
              Regulatory
            </span>
          )}
          {labelCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              {labelCount} labels
            </span>
          )}
          {violationCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
              {violationCount} violations
            </span>
          )}
          {performanceCount > 0 && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              {performanceCount} analytics
            </span>
          )}
        </div>
      </div>

      {/* Summary Stats */}
      {stopData.stop_id_summary && (
        <div className="mt-3 grid grid-cols-4 gap-4 text-sm">
          <div className="text-center">
            <div className="text-gray-600">Stop Topology</div>
            <div className="font-medium">
              {stopData.stop_id_summary.stop_topology?.labels_by_type?.parent_station || 0} labels
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-600">Direction Topo</div>
            <div className="font-medium">
              {stopData.stop_id_summary.direction_topology?.labels_by_type?.direction_id || 0} labels
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-600">Regulatory</div>
            <div className="font-medium">
              {stopData.stop_id_summary.regulatory_stops?.regulatory_stop_ids || 0} stops
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-600">Performance</div>
            <div className="font-medium">
              {stopData.stop_id_summary.performance?.available_performace_analytics || 0} analytics
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StopSequence;