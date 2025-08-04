import React, { useState } from 'react';
import { DirectionData, StopIdData } from '../contexts/DataInterfaces';
import CompactHistogram from './CompactHistogram';

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
        {sortedStops.map((stopIdData) => (
          <StopIdCard 
            key={stopIdData.stop_id}
            stopIdData={stopIdData}
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
  stopIdData: any; // StopIdData with position
  routeId: string;
  directionId: string;
  globalData: any;
}

const StopIdCard: React.FC<StopIdCardProps> = ({ 
  stopIdData, 
  routeId, 
  directionId, 
  globalData 
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [selectedTimeType, setSelectedTimeType] = useState<string>('scheduled');

  const isRegulatory = stopIdData.stop_id_label_keys?.some((key: string) => 
    key.includes('regulatory_stops')
  );

  const labelCount = stopIdData.stop_id_label_keys?.length || 0;
  const violationCount = stopIdData.stop_id_violation_keys?.length || 0;
  const performanceCount = stopIdData.stop_id_performance_keys?.length || 0;

  // Get available time types from globalData
  const availableTimeTypes = globalData?.time_types || ['scheduled'];

  // Get the actual data for the selected time type
  const getDataForTimeType = (timeType: string) => {
    // Get labels (these don't change by time type)
    const labels = (stopIdData.stop_id_label_keys || [])
      .map((key: string) => globalData?.labels?.[key])
      .filter(Boolean);
    
    // Get violations (these don't change by time type)
    const violations = (stopIdData.stop_id_violation_keys || [])
      .map((key: string) => globalData?.violations?.[key])
      .filter(Boolean);
    
    // Find performance data for this specific time type
    const performanceKeys = stopIdData.stop_id_performance_keys || [];
    const performanceKey = performanceKeys.find((key: string) => 
      key.includes(`_${timeType}_`) || key.includes(timeType)
    ) || performanceKeys[0]; // Fallback to first available
    
    const performanceData = performanceKey ? globalData?.performance?.[performanceKey] : null;

    return {
      labels,
      violations,
      performanceData
    };
  };

  const currentData = getDataForTimeType(selectedTimeType);

  return (
    <div className={`border rounded-lg transition-all ${
      isRegulatory 
        ? 'border-amber-200 bg-amber-50' 
        : 'border-gray-200 bg-white hover:bg-gray-50'
    }`}>
      {/* Stop Header */}
      <div className="p-4">
        <div className="flex items-center justify-between">
          {/* Stop Info */}
          <div className="flex items-center space-x-4">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
              isRegulatory 
                ? 'bg-amber-200 text-amber-800' 
                : 'bg-blue-100 text-blue-800'
            }`}>
              {stopIdData.position}
            </div>
            
            <div>
              <h5 className="font-medium text-gray-900">{stopIdData.stop_name}</h5>
              <p className="text-sm text-gray-600">
                Stop ID: {stopIdData.stop_id} • Parent: {stopIdData.parent_station}
              </p>
            </div>
          </div>

          {/* Badges and Toggle */}
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
            
            {/* Show/Hide Details Button */}
            {(labelCount > 0 || violationCount > 0 || performanceCount > 0) && (
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 focus:outline-none transition-colors"
              >
                {showDetails ? 'Hide Details' : 'Show Details'}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Time Type Selector - shown when details are expanded */}
      {showDetails && (
        <div className="px-4 pb-2">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Time Type:</label>
            <select
              value={selectedTimeType}
              onChange={(e) => setSelectedTimeType(e.target.value)}
              className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {availableTimeTypes.map((timeType: string) => (
                <option key={timeType} value={timeType}>
                  {timeType.replace('_', ' ').toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Expanded Details */}
      {showDetails && (
        <div className="border-t border-gray-200 p-4 bg-white">
          <StopDetailsView 
            labels={currentData.labels}
            violations={currentData.violations}
            performanceData={currentData.performanceData}
          />
        </div>
      )}
    </div>
  );
};

// Component that shows the detailed data for a stop
const StopDetailsView: React.FC<{
  labels: any[];
  violations: any[];
  performanceData: any;
}> = ({ labels, violations, performanceData }) => {
  
  return (
    <div className="space-y-6">
      
      {/* Performance Details */}
      {performanceData?.analytics && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Performance Analysis</h4>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Punctuality Breakdown */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h5 className="text-xs font-medium text-gray-600 mb-3">PUNCTUALITY DISTRIBUTION</h5>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-green-600">On Time</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-yellow-600">Too Early</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.too_early}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-red-600">Too Late</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.punctuality_distribution.percentages.too_late}%</span>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h5 className="text-xs font-medium text-gray-600 mb-3">DELAY STATISTICS</h5>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Mean</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.basic_statistics.mean_delay.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Median</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.basic_statistics.median_delay}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Sample Size</span>
                  <span className="text-sm font-medium">{performanceData.analytics.punctuality.sample_size.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Compact Histogram */}
            <CompactHistogram performanceData={performanceData} />
          </div>
        </div>
      )}

      {/* Labels */}
      {labels && labels.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Labels ({labels.length})</h4>
          <div className="space-y-2">
            {labels.map((label, index) => (
              <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {label.label_type}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{label.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Violations */}
      {violations && violations.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Violations ({violations.length})</h4>
          <div className="space-y-2">
            {violations.map((violation, index) => {
              const getSeverityColor = (severity: number) => {
                if (severity >= 5) return 'bg-red-100 text-red-800 border-red-200'
                if (severity >= 3) return 'bg-orange-100 text-orange-800 border-orange-200'
                return 'bg-yellow-100 text-yellow-800 border-yellow-200'
              }

              return (
                <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          {violation.violation_type}
                        </span>
                        {violation.severity && (
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(violation.severity)}`}>
                            Severity {violation.severity}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-700">{violation.description}</p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* No Additional Data */}
      {(!labels || labels.length === 0) && (!violations || violations.length === 0) && !performanceData?.analytics && (
        <div className="text-center py-6 text-gray-500 text-sm">
          No detailed data found for this combination
        </div>
      )}
    </div>
  )
};

export default StopSequence;