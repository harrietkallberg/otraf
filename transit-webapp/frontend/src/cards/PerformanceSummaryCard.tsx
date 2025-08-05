import React, { useMemo } from 'react';
import { PerformanceSummary } from '../shared/types';

// Hook to aggregate labels and violations by filtering global data based on context
const useAggregatedIssues = (data: PerformanceSummary | null, globalData: any, context: { type: 'route' | 'stop', id: string }) => {
  return useMemo(() => {
    if (!data || !globalData) {
      return {
        labels: [],
        violations: []
      };
    }

    // Convert labels/violations to arrays if needed
    const labelsArray = Array.isArray(globalData.labels) ? globalData.labels : Object.values(globalData.labels || {});
    const violationsArray = Array.isArray(globalData.violations) ? globalData.violations : Object.values(globalData.violations || {});

    let relevantLabels: any[] = [];
    let relevantViolations: any[] = [];

    if (context.type === 'route') {
      // Filter by route_id for route-level data
      relevantLabels = labelsArray.filter((label: any) => 
        label.route_id === context.id || 
        label.entity_key?.includes(`_${context.id}_`)
      );
      
      relevantViolations = violationsArray.filter((violation: any) => 
        violation.route_id === context.id || 
        violation.entity_key?.includes(`_${context.id}_`)
      );
      
    } else if (context.type === 'stop') {
      // For stop-level data, get all stop_ids associated with this parent station
      const parentStationData = globalData.stops?.[context.id];
      const associatedStopIds = parentStationData?.stop_ids || [context.id];
      
      relevantLabels = labelsArray.filter((label: any) => 
        // Match parent station
        label.parent_station === context.id ||
        label.entity_key?.includes(`_${context.id}_`) ||
        // Match any associated stop_id
        associatedStopIds.some((stopId: string) => 
          label.stop_id === stopId || 
          label.entity_key?.includes(`_${stopId}_`)
        )
      );
      
      relevantViolations = violationsArray.filter((violation: any) => 
        // Match parent station
        violation.parent_station === context.id ||
        violation.entity_key?.includes(`_${context.id}_`) ||
        // Match any associated stop_id
        associatedStopIds.some((stopId: string) => 
          violation.stop_id === stopId || 
          violation.entity_key?.includes(`_${stopId}_`)
        )
      );
    }

    return {
      labels: relevantLabels,
      violations: relevantViolations
    };
  }, [data, globalData, context]);
};

// Issue Breakdown component
interface IssueBreakdownProps {
  labels: any[];
  violations: any[];
  performanceData?: any;
}

const IssueBreakdown: React.FC<IssueBreakdownProps> = ({ 
  labels, 
  violations, 
  performanceData 
}) => {
  // Calculate issue metrics by severity level (1-5)
  const severity5Violations = violations.filter(v => v.severity === 5).length;
  const severity4Violations = violations.filter(v => v.severity === 4).length;
  const severity3Violations = violations.filter(v => v.severity === 3).length;
  const severity2Violations = violations.filter(v => v.severity === 2).length;
  const severity1Violations = violations.filter(v => v.severity === 1).length;
 
  // Performance issues
  const hasPerformanceIssues = performanceData?.overall_on_time_rate < 80;
  const hasHighDelays = performanceData?.average_departure_delay > 120;
  
  const totalIssues = violations.length + (hasPerformanceIssues ? 1 : 0) + (hasHighDelays ? 1 : 0);

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h4 className="text-sm font-medium text-gray-700 mb-3">Issue Summary</h4>
      <div className="space-y-3">
        
        {/* Severity 5 - Critical */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-600"></div>
            <span className="text-sm text-gray-600">Critical (Severity 5)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity5Violations > 0 ? 'text-red-600' : 'text-gray-400'
          }`}>
            {severity5Violations}
          </span>
        </div>

        {/* Severity 4 - High */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span className="text-sm text-gray-600">High (Severity 4)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity4Violations > 0 ? 'text-red-500' : 'text-gray-400'
          }`}>
            {severity4Violations}
          </span>
        </div>

        {/* Severity 3 - Medium */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-orange-400"></div>
            <span className="text-sm text-gray-600">Medium (Severity 3)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity3Violations > 0 ? 'text-orange-600' : 'text-gray-400'
          }`}>
            {severity3Violations}
          </span>
        </div>

        {/* Severity 2 - Low */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span className="text-sm text-gray-600">Low (Severity 2)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity2Violations > 0 ? 'text-yellow-600' : 'text-gray-400'
          }`}>
            {severity2Violations}
          </span>
        </div>

        {/* Severity 1 - Minimal */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-yellow-300"></div>
            <span className="text-sm text-gray-600">Minimal (Severity 1)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity1Violations > 0 ? 'text-yellow-600' : 'text-gray-400'
          }`}>
            {severity1Violations}
          </span>
        </div>

        {/* No Issues Found */}
        {totalIssues === 0 && labels.length === 0 && (
          <div className="text-center py-2">
            <span className="text-sm text-green-600 font-medium">✓ No Issues Found</span>
          </div>
        )}
      </div>
    </div>
  );
};

interface PerformanceSummaryProps {
  title: string;
  data: PerformanceSummary | null;
  globalData: any;
  context: { type: 'route' | 'stop', id: string };
}

const PerformanceSummaryCard: React.FC<PerformanceSummaryProps> = ({ 
  title, 
  data, 
  globalData,
  context 
}) => {
  // Always call hooks at the top level
  const { labels, violations } = useAggregatedIssues(data, globalData, context);

  if (!data) {
    return (
      <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="text-gray-500">No performance data available</div>
      </div>
    );
  }
  
  return (
    <div className="bg-white shadow-sm rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-6">{title}</h3>
      
      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        
        {/* On Time Rate */}
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 bg-green-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div className="text-2xl font-bold text-green-600">{data.overall_on_time_rate.toFixed(1)}%</div>
          <div className="text-sm font-medium text-gray-700">On Time</div>
          <div className="text-xs text-gray-500 mt-1">Within schedule window</div>
        </div>

        {/* Too Early Rate */}
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 bg-yellow-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="text-2xl font-bold text-yellow-600">{data.overall_too_early_rate.toFixed(1)}%</div>
          <div className="text-sm font-medium text-gray-700">Too Early</div>
          <div className="text-xs text-gray-500 mt-1">Ahead of schedule</div>
        </div>

        {/* Too Late Rate */}
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 bg-red-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <div className="text-2xl font-bold text-red-600">{data.overall_too_late_rate.toFixed(1)}%</div>
          <div className="text-sm font-medium text-gray-700">Too Late</div>
          <div className="text-xs text-gray-500 mt-1">Behind schedule</div>
        </div>

        {/* Average Delay */}
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 bg-blue-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div className="text-2xl font-bold text-blue-600">{data.average_departure_delay.toFixed(0)}s</div>
          <div className="text-sm font-medium text-gray-700">Avg Delay</div>
          <div className="text-xs text-gray-500 mt-1">Mean departure delay</div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Canonical Share */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium text-gray-700">Canonical Share</h4>
                <p className="text-xs text-gray-500 mt-1">
                  Proportion following expected route pattern
                </p>
              </div>
              <div className="text-right">
                <div className="text-xl font-bold text-gray-900">
                  {(data.canonical_share * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            
            {/* Progress bar */}
            <div className="mt-3">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${data.canonical_share * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
          
          {/* Issue Breakdown */}
          <IssueBreakdown 
            labels={labels}
            violations={violations}
            performanceData={data}
          />
        </div>
      </div>
    </div>
  );
};

export default PerformanceSummaryCard;