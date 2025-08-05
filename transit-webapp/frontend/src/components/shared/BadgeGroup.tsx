// src/components/shared/BadgeGroup.tsx
import React from 'react';
import { Badge } from './Badge';
import { StopIdData, StopIdDataWithPosition } from '../../shared/types';

interface BadgeGroupProps {
  stopIdData: StopIdData | StopIdDataWithPosition; // Accept both types
  performanceData?: any;
  size?: 'sm' | 'md';
  includeRegulatory?: boolean;
}

export const BadgeGroup: React.FC<BadgeGroupProps> = ({ 
  stopIdData, 
  performanceData, 
  size = 'md',
  includeRegulatory = true 
}) => {
  const isRegulatory = stopIdData.stop_id_label_keys.some((key: string) => 
    key.includes('regulatory_stops')
  );
  
  const labelCount = stopIdData.stop_id_label_keys.length;
  const violationCount = stopIdData.stop_id_violation_keys.length;
  const performanceCount = stopIdData.stop_id_performance_keys.length;

  return (
    <div className="flex items-center space-x-2">
      {includeRegulatory && isRegulatory && (
        <Badge count={0} type="regulatory" size={size} customText="Regulatory" />
      )}
      {labelCount > 0 && (
        <Badge count={labelCount} type="labels" size={size} />
      )}
      {violationCount > 0 && (
        <Badge count={violationCount} type="violations" size={size} />
      )}
      {performanceCount > 0 && (
        <Badge count={performanceCount} type="analytics" size={size} />
      )}
      {performanceData?.analytics?.punctuality && (
        <Badge 
          count={0} 
          type="analytics" 
          size={size}
          customText={`${performanceData.analytics.punctuality.punctuality_distribution.percentages.on_time}% on time`}
        />
      )}
    </div>
  );
};