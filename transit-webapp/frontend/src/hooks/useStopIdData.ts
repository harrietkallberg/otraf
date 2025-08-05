
// src/hooks/useStopIdData.ts
import { useMemo } from 'react';
import { StopIdData, StopIdDataWithPosition } from '../shared/types';

export const useStopIdData = (stopIdData: StopIdData | StopIdDataWithPosition, globalData: any, timeType: string) => {
  return useMemo(() => {
    const labels = stopIdData.stop_id_label_keys
      .map((key: string) => globalData?.labels?.[key])
      .filter(Boolean);
    
    const violations = stopIdData.stop_id_violation_keys
      .map((key: string) => globalData?.violations?.[key])
      .filter(Boolean);
    
    const performanceKeys = stopIdData.stop_id_performance_keys || [];
    const performanceKey = performanceKeys.find((key: string) => 
      key.includes(`_${timeType}_`) || key.includes(timeType)
    ) || performanceKeys[0];
    
    const performanceData = performanceKey ? globalData?.performance?.[performanceKey] : null;

    return {
      labels,
      violations,
      performanceData,
      hasData: labels.length > 0 || violations.length > 0 || !!performanceData
    };
  }, [stopIdData, globalData, timeType]);
};