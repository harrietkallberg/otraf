// src/components/shared/PerformanceDetails.tsx
import React from 'react';
import CompactHistogram from '../../cards/CompactHistogram';

interface PerformanceDetailsProps {
  performanceData: any;
  size?: 'sm' | 'md' | 'lg' | null;
}

export const PerformanceDetails: React.FC<PerformanceDetailsProps> = ({ 
  performanceData, 
  size = 'md' 
}) => {
  if (!performanceData?.analytics) return null;

  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';
  const headerSize = size === 'sm' ? 'text-xs' : 'text-xs';
  const padding = size === 'sm' ? 'p-2' : size === 'md' ? 'p-3' : 'p-4';
  const spacing = size === 'sm' ? 'space-y-1' : 'space-y-2';

  return (
    <div>
      <h4 className={`${textSize} font-medium text-gray-700 mb-3`}>Performance Analysis</h4>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        
        <div className={`bg-gray-50 rounded-lg ${padding}`}>
          <h5 className={`${headerSize} font-medium text-gray-600 mb-2`}>PUNCTUALITY DISTRIBUTION</h5>
          <div className={spacing}>
            <div className="flex justify-between items-center">
              <span className={`${textSize} text-green-600`}>On Time</span>
              <span className={`${textSize} font-medium`}>
                {performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className={`${textSize} text-yellow-600`}>Too Early</span>
              <span className={`${textSize} font-medium`}>
                {performanceData.analytics.punctuality.punctuality_distribution.percentages.too_early}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className={`${textSize} text-red-600`}>Too Late</span>
              <span className={`${textSize} font-medium`}>
                {performanceData.analytics.punctuality.punctuality_distribution.percentages.too_late}%
              </span>
            </div>
          </div>
        </div>

        <div className={`bg-gray-50 rounded-lg ${padding}`}>
          <h5 className={`${headerSize} font-medium text-gray-600 mb-2`}>DELAY STATISTICS</h5>
          <div className={spacing}>
            <div className="flex justify-between items-center">
              <span className={`${textSize} text-gray-600`}>Mean</span>
              <span className={`${textSize} font-medium`}>
                {performanceData.analytics.punctuality.basic_statistics.mean_delay.toFixed(1)}s
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className={`${textSize} text-gray-600`}>Median</span>
              <span className={`${textSize} font-medium`}>
                {performanceData.analytics.punctuality.basic_statistics.median_delay}s
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className={`${textSize} text-gray-600`}>Sample Size</span>
              <span className={`${textSize} font-medium`}>
                {performanceData.analytics.punctuality.sample_size.toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        <CompactHistogram performanceData={performanceData} />
      </div>
    </div>
  );
};