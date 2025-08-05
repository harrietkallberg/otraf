// src/components/cards/StopSequence.tsx (SIMPLIFIED)
import React from 'react';
import { DirectionData } from '../shared/types';
import { StopCard } from '../components/shared';

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
        {sortedStops.map((stopIdData) => (
          <StopCard 
            key={stopIdData.stop_id}
            stopIdData={stopIdData}
            globalData={globalData}
            size="md"
            showTimeSelector={true}
          />
        ))}
      </div>
    </div>
  );
};

export default StopSequence;