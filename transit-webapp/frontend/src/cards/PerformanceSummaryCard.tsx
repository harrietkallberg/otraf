import React from 'react';
import { PerformanceSummary } from '../contexts/DataInterfaces';

interface PerformanceSummaryProps {
  title: string;
  data: PerformanceSummary | null;
}

const PerformanceSummaryCard: React.FC<PerformanceSummaryProps> = ({ title, data }) => {
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

          {/* Performance Summary */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Quick Summary</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Best performing:</span>
                <span className="font-medium text-green-600">
                  {data.overall_on_time_rate > Math.max(data.overall_too_early_rate, data.overall_too_late_rate) 
                    ? 'On Time' 
                    : data.overall_too_early_rate > data.overall_too_late_rate 
                      ? 'Too Early' 
                      : 'Too Late'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Delay trend:</span>
                <span className={`font-medium ${
                  data.average_departure_delay > 60 
                    ? 'text-red-600' 
                    : data.average_departure_delay > 30 
                      ? 'text-yellow-600' 
                      : 'text-green-600'
                }`}>
                  {data.average_departure_delay > 60 
                    ? 'High delays' 
                    : data.average_departure_delay > 30 
                      ? 'Moderate delays' 
                      : 'Low delays'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Route adherence:</span>
                <span className={`font-medium ${
                  data.canonical_share > 0.9 
                    ? 'text-green-600' 
                    : data.canonical_share > 0.8 
                      ? 'text-yellow-600' 
                      : 'text-red-600'
                }`}>
                  {data.canonical_share > 0.9 
                    ? 'Excellent' 
                    : data.canonical_share > 0.8 
                      ? 'Good' 
                      : 'Needs improvement'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceSummaryCard;