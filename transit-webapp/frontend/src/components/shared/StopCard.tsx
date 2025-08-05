// src/components/shared/StopCard.tsx
import React, { useState } from 'react';
import { StopIdData, StopIdDataWithPosition } from '../../shared/types';
import { BadgeGroup } from './BadgeGroup';
import { TimeTypeSelector } from './TimeTypeSelector';
import { StopDetailsView } from './StopDetailsView';
import { useStopIdData  } from '../../hooks/useStopIdData';

interface StopCardProps {
  stopIdData: StopIdData | StopIdDataWithPosition; // Accept both types
  globalData: any;
  initialTimeType?: string;
  externalTimeType?: string;
  onTimeTypeChange?: (timeType: string) => void;
  size?: 'sm' | 'md' | 'lg';
  showTimeSelector?: boolean;
  alwaysExpanded?: boolean;
  positionBadgeSize?: 'sm' | 'md';
}

export const StopCard: React.FC<StopCardProps> = ({
  stopIdData,
  globalData,
  initialTimeType = 'scheduled',
  externalTimeType,
  onTimeTypeChange,
  size = 'md',
  showTimeSelector = true,
  alwaysExpanded = false,
  positionBadgeSize = 'md'
}) => {
  const [internalTimeType, setInternalTimeType] = useState(initialTimeType);
  const [showDetails, setShowDetails] = useState(alwaysExpanded);
  
  const currentTimeType = externalTimeType || internalTimeType;
  const availableTimeTypes = globalData?.time_types || ['scheduled'];
  
  const { labels, violations, performanceData, hasData } = useStopIdData(
    stopIdData, 
    globalData, 
    currentTimeType
  );

  const handleTimeTypeChange = (newTimeType: string) => {
    if (onTimeTypeChange) {
      onTimeTypeChange(newTimeType);
    } else {
      setInternalTimeType(newTimeType);
    }
  };

  const isRegulatory = stopIdData.stop_id_label_keys.some((key: string) => 
    key.includes('regulatory_stops')
  );

  const badgeSize = positionBadgeSize === 'sm' ? 'w-6 h-6 text-xs' : 'w-8 h-8 text-sm';
  const textSize = size === 'sm' ? 'text-sm' : 'text-base';
  const subTextSize = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <div className={`border rounded-lg transition-all ${
      isRegulatory 
        ? 'border-amber-200 bg-amber-50' 
        : 'border-gray-200 bg-white hover:bg-gray-50'
    }`}>
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className={`${badgeSize} rounded-full flex items-center justify-center font-bold ${
              isRegulatory 
                ? 'bg-amber-200 text-amber-800' 
                : 'bg-blue-100 text-blue-800'
            }`}>
              {'position' in stopIdData ? stopIdData.position : '?'}
            </div>
            
            <div>
              <h5 className={`font-medium text-gray-900 ${textSize}`}>{stopIdData.stop_name}</h5>
              <p className={`text-gray-600 ${subTextSize}`}>
                Stop ID: {stopIdData.stop_id} • Parent: {stopIdData.parent_station}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <BadgeGroup 
              stopIdData={stopIdData} 
              performanceData={performanceData}
              size={size === 'sm' ? 'sm' : 'md'}
            />
            
            {!alwaysExpanded && hasData && (
              <button
                onClick={() => setShowDetails(!showDetails)}
                className={`inline-flex items-center font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors ${subTextSize}`}
              >
                {showDetails ? 'Hide Details' : 'Show Details'}
              </button>
            )}
          </div>
        </div>
      </div>

      {(showDetails || alwaysExpanded) && showTimeSelector && (
        <div className="px-4 pb-2">
          <TimeTypeSelector
            selectedTimeType={currentTimeType}
            availableTimeTypes={availableTimeTypes}
            onTimeTypeChange={handleTimeTypeChange}
            size={size === 'sm' ? 'sm' : 'md'}
          />
        </div>
      )}

      {(showDetails || alwaysExpanded) && (
        <div className="border-t border-gray-200 p-4 bg-white">
          <StopDetailsView 
            labels={labels}
            violations={violations}
            performanceData={performanceData}
            size={size}
          />
        </div>
      )}
    </div>
  );
};
