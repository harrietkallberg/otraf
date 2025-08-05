import React, { useContext } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { GlobalDataContext } from '../contexts/GlobalDataContext';

// Register necessary chart components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface PunctualityFlowChartProps {
  routeId: string;
  directionId: string;
  direction: any;
  selectedTimeType: string;
  selectedMetric: string;
}

const PunctualityFlowChart: React.FC<PunctualityFlowChartProps> = ({ 
  routeId, 
  directionId, 
  direction, 
  selectedTimeType, 
  selectedMetric 
}) => {
  const globalData = useContext(GlobalDataContext);

  if (!globalData || !direction.stop_ids_in_direction) {
    return <div className="text-gray-500 text-sm">No performance data available</div>;
  }

  // Extract canonical pattern and sort by position
  const sortedStops = Object.entries(direction.stop_ids_in_direction)
    .sort(([a], [b]) => parseInt(a) - parseInt(b))
    .map(([position, stop]: [string, any]) => ({
      position: parseInt(position),
      stopIdData: stop,
      ...stop
    }));

  // Collect performance data for each stop
  const performanceData = sortedStops.map(stop => {
    // Find the performance key manually (similar to the hook logic)
    const performanceKeys = stop.stopIdData.stop_id_performance_keys || [];
    const performanceKey = performanceKeys.find((key: string) => 
      key.includes(`_${selectedTimeType}_`) || key.includes(selectedTimeType)
    ) || performanceKeys[0];
    
    const stopPerformanceData = performanceKey ? globalData?.performance?.[performanceKey] : null;

    let value = 0;
    if (stopPerformanceData?.analytics?.punctuality?.punctuality_distribution?.percentages) {
      value = stopPerformanceData.analytics.punctuality.punctuality_distribution.percentages[selectedMetric] || 0;
    }

    // Check if this is a regulatory stop by looking at the stop_id_summary
    const isRegulatory = stop.stopIdData?.stop_id_summary?.regulatory_stops?.regulatory_stop_ids > 0;

    return {
      ...stop,
      value,
      hasData: !!stopPerformanceData,
      isRegulatory
    };
  });

  const maxValue = Math.max(...performanceData.map(d => d.value), 100);

  const getColor = (value: number, metric: string, isRegulatory: boolean = false) => {
    const intensity = Math.min(value / maxValue, 1);
    
    // Regular colors for non-regulatory stops
    switch (metric) {
      case 'on_time':
        return `rgba(34, 197, 94, ${0.3 + intensity * 0.7})`; // Green
      case 'too_late':
        return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`; // Red
      case 'too_early':
        return `rgba(245, 158, 11, ${0.2 + intensity * 0.7})`; // Yellow
      default:
        return `rgba(156, 163, 175, ${0.2 + intensity * 0.7})`; // Gray
    }
  };

  const getBorderColor = (value: number, metric: string, isRegulatory: boolean = false) => {
    if (!isRegulatory) return 'transparent';
    // Return darker versions of the bar colors for regulatory stops
   switch (metric) {
      case 'on_time':
        return 'rgba(37, 151, 79, 1)'; // Darker green
      case 'too_late':
        return 'rgba(201, 75, 75, 1)'; // Darker red
      case 'too_early':
        return 'rgba(255, 147, 64, 1)'; // Darker orange
      default:
        return 'rgba(88, 101, 122, 1)'; // Darker gray
    }
  };

  const getBorderWidth = (isRegulatory: boolean) => {
    return isRegulatory ? 2 : 0; // No border for regulatory stops
  };

  const getMetricDisplayName = (metric: string) => {
    switch (metric) {
      case 'on_time':
        return 'ON TIME';
      case 'too_late':
        return 'TOO LATE';
      case 'too_early':
        return 'TOO EARLY';
      default:
        return metric.toUpperCase();
    }
  };

  const getTimeTypeDisplayName = (timeType: string) => {
    switch (timeType) {
      case 'am_rush':
        return 'AM RUSH';
      case 'pm_rush':
        return 'PM RUSH';
      case 'day':
        return 'DAY';
      case 'night':
        return 'NIGHT';
      case 'weekend':
        return 'WEEKEND';
      default:
        return timeType.toUpperCase();
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex flex-col space-y-4">
        {/* Chart Header */}
        <div className="text-center">
          <h5 className="text-sm font-medium text-gray-700">
            {getMetricDisplayName(selectedMetric)} - {getTimeTypeDisplayName(selectedTimeType)}
          </h5>
        </div>

        {/* Chart */}
        <div className="relative">
          <div className="flex items-end justify-between space-x-1 h-48">
            {performanceData.map((stop, index) => (
              <div key={stop.stop_id} className="flex flex-col items-center flex-1 group">
                {/* Bar */}
                <div className="relative w-full flex flex-col justify-end h-40">
                  <div
                    className="w-full rounded-t transition-all duration-200 group-hover:opacity-80"
                    style={{
                      height: `${(stop.value / maxValue) * 100}%`,
                      backgroundColor: getColor(stop.value, selectedMetric, stop.isRegulatory),
                      border: `${getBorderWidth(stop.isRegulatory)}px solid ${getBorderColor(stop.value, selectedMetric, stop.isRegulatory)}`,
                      minHeight: stop.hasData ? '4px' : '2px'
                    }}
                  />

                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 whitespace-nowrap">
                    <div className="font-medium">{stop.stop_name}</div>
                    <div className="text-center">
                      {stop.hasData ? `${stop.value.toFixed(1)}%` : 'No data'}
                    </div>
                    <div className="text-center text-gray-300">
                      Position {stop.position}
                    </div>
                    {stop.isRegulatory && (
                      <div className="text-center text-orange-500 font-bold">
                        Regulatory Stop
                      </div>
                    )}
                  </div>
                </div>

                {/* Position label with fixed height container */}
                <div className="h-6 flex items-center justify-center mt-1">
                  <div 
                    className={`text-xs ${
                      stop.isRegulatory 
                        ? 'font-bold text-orange-500 bg-orange-100 px-2 py-1 rounded-md' 
                        : 'font-medium text-gray-600'
                    }`}
                  >
                    {stop.position}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 h-40 flex flex-col justify-between text-xs text-gray-500 -ml-10">
            <span>{maxValue.toFixed(0)}%</span>
            <span>{(maxValue * 0.75).toFixed(0)}%</span>
            <span>{(maxValue * 0.5).toFixed(0)}%</span>
            <span>{(maxValue * 0.25).toFixed(0)}%</span>
            <span>0%</span>
          </div>
        </div>

        {/* X-axis label */}
        <div className="text-center text-xs text-gray-500 mt-2">Stop Position in Route</div>

        {/* Legend */}
        <div className="flex justify-center items-center space-x-4 text-xs text-gray-600">
          <div className="flex items-center space-x-1">
            <div
              className="w-3 h-3 rounded border border-gray-300"
              style={{ backgroundColor: getColor(maxValue, selectedMetric, false) }}
            />
            <span>Regular Stop</span>
          </div>
          <div className="flex items-center space-x-1">
            <div
              className="w-3 h-3 rounded border-2 border-gray-800 shadow"
              style={{ backgroundColor: getColor(maxValue, selectedMetric, true) }}
            />
            <span>Regulatory Stop</span>
          </div>
          <span>•</span>
          <span>Hover for details</span>
        </div>

        {/* Summary */}
        <div className="text-center text-sm text-gray-600">
          Showing {performanceData.filter((d) => d.hasData).length} of {performanceData.length} stops with data
          {performanceData.filter((d) => d.isRegulatory).length > 0 && (
            <span className="ml-2 text-yellow-700 font-medium">
              ({performanceData.filter((d) => d.isRegulatory).length} regulatory)
            </span>
          )}
        </div>

        {/* Performance Summary */}
        {performanceData.filter((d) => d.hasData).length > 0 && (
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-sm font-medium text-gray-700">Average</div>
                <div className="text-lg font-bold">
                  {(performanceData.filter((d) => d.hasData).reduce((sum, d) => sum + d.value, 0) / performanceData.filter((d) => d.hasData).length).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700">Best Stop</div>
                <div className="text-lg font-bold text-green-600">
                  {Math.max(...performanceData.filter((d) => d.hasData).map((d) => d.value)).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700">Worst Stop</div>
                <div className="text-lg font-bold text-red-600">
                  {Math.min(...performanceData.filter((d) => d.hasData).map((d) => d.value)).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PunctualityFlowChart;