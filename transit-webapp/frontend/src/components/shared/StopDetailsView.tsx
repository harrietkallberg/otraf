// src/components/shared/StopDetailsView.tsx
import React from 'react';
import { PerformanceDetails } from './PerformanceDetails';
import { LabelsViolationsDisplay } from './LabelsViolationsDisplay';

interface StopDetailsViewProps {
  labels: any[];
  violations: any[];
  performanceData: any;
  size?: 'sm' | 'md' | 'lg';
}

export const StopDetailsView: React.FC<StopDetailsViewProps> = ({ 
  labels, 
  violations, 
  performanceData, 
  size = 'md' 
}) => {
  const spacing = size === 'sm' ? 'space-y-3' : size === 'md' ? 'space-y-4' : 'space-y-6';
  
  return (
    <div className={spacing}>
      <PerformanceDetails performanceData={performanceData} size={size} />
      <LabelsViolationsDisplay labels={labels} violations={violations} size={size} />
    </div>
  );
};

export default StopDetailsView