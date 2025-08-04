import React, { useState } from 'react';
import { DirectionData } from '../contexts/DataInterfaces';
import ResultTile from './ResultTile'; // Import the entire ResultTile component

interface LogSpecificationsProps {
  direction: DirectionData;
  routeId: string;
  directionId: string;
  globalData: any;
}

interface ResultCombination {
  routeId: string;
  directionId: string;
  stopId: string;
  timeType: string;
  stopName: string;
  routeName: string;
  performanceKey: string;
  hasPerformance: boolean;
  labelCount: number;
  violationCount: number;
  performanceData?: any;
  labels: any[];
  violations: any[];
}

const LogSpecifications: React.FC<LogSpecificationsProps> = ({ 
  direction, 
  routeId, 
  directionId, 
  globalData 
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [selectedTimeType, setSelectedTimeType] = useState<string>('scheduled');

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  };

  const stopIdsInDirection = direction.stop_ids_in_direction;
  
  if (!stopIdsInDirection) {
    return <div className="text-gray-500 text-sm">No log specifications available.</div>;
  }

  // Sort stops by position
  const sortedStops = Object.entries(stopIdsInDirection)
    .sort(([a], [b]) => parseInt(a) - parseInt(b))
    .map(([position, stopData]: [string, any]) => ({
      position: parseInt(position),
      ...stopData
    }));

  // Get available time types from globalData
  const availableTimeTypes = globalData?.time_types || ['scheduled'];

  // Helper function to create ResultCombination for each stop and time type
  const createResultCombinations = (): ResultCombination[] => {
    const combinations: ResultCombination[] = [];

    sortedStops.forEach((stop) => {
      // For each time type, create a result combination
      availableTimeTypes.forEach((timeType: string) => {
        const labels = (stop.stop_id_label_keys || [])
          .map((key: string) => globalData?.labels?.[key])
          .filter(Boolean);
        
        const violations = (stop.stop_id_violation_keys || [])
          .map((key: string) => globalData?.violations?.[key])
          .filter(Boolean);
        
        // Find performance data for this specific time type if available
        const performanceKeys = stop.stop_id_performance_keys || [];
        const performanceKey = performanceKeys.find((key: string) => 
          key.includes(`_${timeType}_`) || key.includes(timeType)
        ) || performanceKeys[0]; // Fallback to first available
        
        const performanceData = performanceKey ? globalData?.performance?.[performanceKey] : null;

        // Only include combinations that have some data (performance, labels, or violations)
        if (performanceData || labels.length > 0 || violations.length > 0) {
          combinations.push({
            routeId,
            directionId,
            stopId: stop.stop_id,
            timeType,
            stopName: stop.stop_name,
            routeName: globalData?.routes?.[routeId]?.route_short_name || routeId,
            performanceKey: performanceKey || '',
            hasPerformance: !!performanceData,
            labelCount: labels.length,
            violationCount: violations.length,
            performanceData,
            labels,
            violations
          });
        }
      });
    });

    return combinations;
  };

  const resultCombinations = createResultCombinations();

  // Filter combinations by selected time type
  const filteredCombinations = selectedTimeType === 'all' 
    ? resultCombinations 
    : resultCombinations.filter(combo => combo.timeType === selectedTimeType);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-lg font-semibold text-gray-900">Log Specifications</h4>
        
        {/* Time Type Filter */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700">Time Type:</label>
          <select
            value={selectedTimeType}
            onChange={(e) => setSelectedTimeType(e.target.value)}
            className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Types</option>
            {availableTimeTypes.map((timeType: string) => (
              <option key={timeType} value={timeType}>
                {timeType.replace('_', ' ').toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="space-y-4">
        
        {/* Summary Statistics */}
        <LogSection
          title={`Summary Statistics (${filteredCombinations.length} combinations)`}
          sectionKey="summary"
          isExpanded={expandedSections.has('summary')}
          onToggle={() => toggleSection('summary')}
        >
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {filteredCombinations.reduce((sum, combo) => sum + combo.labelCount, 0)}
              </div>
              <div className="text-sm text-blue-800">Total Labels</div>
            </div>
            <div className="text-center p-3 bg-red-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">
                {filteredCombinations.reduce((sum, combo) => sum + combo.violationCount, 0)}
              </div>
              <div className="text-sm text-red-800">Total Violations</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {filteredCombinations.filter(combo => combo.hasPerformance).length}
              </div>
              <div className="text-sm text-green-800">Performance Records</div>
            </div>
            <div className="text-center p-3 bg-amber-50 rounded-lg">
              <div className="text-2xl font-bold text-amber-600">
                {filteredCombinations.filter(combo => 
                  combo.performanceData?.analytics?.is_regulatory_stop
                ).length}
              </div>
              <div className="text-sm text-amber-800">Regulatory Stops</div>
            </div>
          </div>
        </LogSection>

        {/* Result Tiles Section */}
        <LogSection
          title="Detailed Results"
          sectionKey="results"
          isExpanded={expandedSections.has('results')}
          onToggle={() => toggleSection('results')}
        >
          <div className="space-y-4">
            {filteredCombinations.length > 0 ? (
              filteredCombinations.map((combination, index) => (
                <ResultTile
                  key={`${combination.stopId}_${combination.timeType}`}
                  result={combination}
                  index={index}
                  globalData={globalData}
                />
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                No data found for the selected time type.
              </div>
            )}
          </div>
        </LogSection>
      </div>
    </div>
  );
};

interface LogSectionProps {
  title: string;
  sectionKey: string;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

const LogSection: React.FC<LogSectionProps> = ({ 
  title, 
  isExpanded, 
  onToggle, 
  children 
}) => {
  return (
    <div className="border border-gray-200 rounded-lg">
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 text-left flex items-center justify-between bg-gray-50 rounded-t-lg hover:bg-gray-100 transition-colors"
      >
        <h5 className="font-medium text-gray-900">{title}</h5>
        <svg 
          className={`w-5 h-5 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isExpanded && (
        <div className="p-4 border-t border-gray-200">
          {children}
        </div>
      )}
    </div>
  );
};

export default LogSpecifications;
