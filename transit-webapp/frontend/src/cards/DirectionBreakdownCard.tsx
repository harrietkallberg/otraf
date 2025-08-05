import React, { useState } from 'react';
import { DirectionData } from '../shared/types';
import PunctualityFlowChart from './PunctualityFlowChart';
import StopSequence from './StopSequence';

interface DirectionBreakdownCardProps {
  title: string;
  directions: { [directionId: string]: DirectionData };
  routeId: string;
  globalData: any;
}

const DirectionBreakdownCard: React.FC<DirectionBreakdownCardProps> = ({ 
  title, 
  directions, 
  routeId, 
  globalData 
}) => {
  const [showDirections, setShowDirections] = useState(false);

  const toggleDirectionsVisibility = () => {
    setShowDirections(!showDirections);
  };

  return (
    <div className="bg-white shadow-sm rounded-lg border border-gray-200">
      <div className="px-6 py-4 flex justify-between items-center border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        <button
          onClick={toggleDirectionsVisibility}
          className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
        >
          {showDirections ? "Hide Directions" : "Show Directions"}
        </button>
      </div>

      {showDirections && (
        <div className="p-6 space-y-6">
          {Object.entries(directions).map(([directionId, direction]) => (
            <DirectionTile 
              key={directionId}
              directionId={directionId}
              direction={direction}
              routeId={routeId}
              globalData={globalData}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface DirectionTileProps {
  directionId: string;
  direction: DirectionData;
  routeId: string;
  globalData: any;
}

const DirectionTile: React.FC<DirectionTileProps> = ({ 
  directionId, 
  direction, 
  routeId, 
  globalData 
}) => {
  const [showPunctualityFlow, setShowPunctualityFlow] = useState(false);
  const [showStopSequence, setShowStopSequence] = useState(false);
  const [selectedTimeType, setSelectedTimeType] = useState('day');
  const [selectedMetric, setSelectedMetric] = useState('on_time');

  // Calculate markers from direction_summary
  const directionLabels = direction.direction_summary?.direction_topology?.labels_by_type?.direction_id || 0;
  const directionViolations = direction.direction_summary?.direction_topology?.violations_by_type?.direction_id || 0;
  const analytics = direction.direction_summary?.performance?.available_performace_analytics || 0;

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      {/* Direction Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-medium text-gray-900">
            Direction {directionId}
          </h3>
          
          {/* Markers */}
          <div className="flex items-center space-x-3">
            {directionLabels > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {directionLabels} labels
              </span>
            )}
            {directionViolations > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                {directionViolations} violations
              </span>
            )}
            {analytics > 0 && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                {analytics} analytics
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Toggle Controls */}
      <div className="flex flex-wrap gap-4 mb-4">
        <button
          onClick={() => setShowPunctualityFlow(!showPunctualityFlow)}
          className={`inline-flex items-center px-3 py-2 text-sm font-medium rounded-md border transition-colors ${
            showPunctualityFlow 
              ? 'bg-blue-50 text-blue-700 border-blue-200' 
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
          }`}
        >
          {showPunctualityFlow ? 'Hide' : 'Show'} Punctuality Flow
        </button>
        
        <button
          onClick={() => setShowStopSequence(!showStopSequence)}
          className={`inline-flex items-center px-3 py-2 text-sm font-medium rounded-md border transition-colors ${
            showStopSequence 
              ? 'bg-blue-50 text-blue-700 border-blue-200' 
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
          }`}
        >
          {showStopSequence ? 'Hide' : 'Show'} Stop Sequence & Details
        </button>
      </div>

      {/* Time Type Selector (shown when punctuality flow is active) */}
      {showPunctualityFlow && (
        <div className="flex items-center space-x-4 mb-4 p-3 bg-gray-50 rounded-lg">
          <label className="text-sm font-medium text-gray-700">Time Type:</label>
          <select
            value={selectedTimeType}
            onChange={(e) => setSelectedTimeType(e.target.value)}
            className="text-sm border border-gray-300 rounded-md px-3 py-1"
          >
            <option value="am_rush">AM Rush</option>
            <option value="day">Day</option>
            <option value="pm_rush">PM Rush</option>
            <option value="night">Night</option>
            <option value="weekend">Weekend</option>
          </select>
          
          <label className="text-sm font-medium text-gray-700">Metric:</label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="text-sm border border-gray-300 rounded-md px-3 py-1"
          >
            <option value="on_time">On Time</option>
            <option value="too_early">Too Early</option>
            <option value="too_late">Too Late</option>
          </select>
        </div>
      )}

      {/* Content Sections */}
      <div className="space-y-6">
        {/* Punctuality Flow Chart */}
        {showPunctualityFlow && (
          <div>
            <PunctualityFlowChart 
              routeId={routeId}
              directionId={directionId}
              direction={direction}
              selectedTimeType={selectedTimeType}
              selectedMetric={selectedMetric}
            />
          </div>
        )}

        {/* Stop Sequence with integrated detailed logs */}
        {showStopSequence && (
          <div>
            <StopSequence 
              direction={direction}
              routeId={routeId}
              directionId={directionId}
              globalData={globalData}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default DirectionBreakdownCard;