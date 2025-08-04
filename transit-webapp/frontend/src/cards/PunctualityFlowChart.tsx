import React, {useContext} from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js'; // Chart.js dependencies
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
    const globalData = useContext(GlobalDataContext)

  if (!globalData || !direction.stop_ids_in_direction) {
    return <div className="text-gray-500 text-sm">No performance data available</div>;
  }

  // Extract canonical pattern and sort by position
  const sortedStops = Object.entries(direction.stop_ids_in_direction)
    .sort(([a], [b]) => parseInt(a) - parseInt(b))
    .map(([position, stop]: [string, any]) => ({
      position: parseInt(position),
      stopId: stop.stop_id,
      stopName: stop.stop_name,
      isRegulatory: stop.regulatory || false  // Assuming `regulatory` field indicates regulatory stops
    }));

  // Collect performance data for each stop
  const performanceData = sortedStops.map(stop => {
    const performanceKey = `performance_${routeId}_direction_id_stop_id_time_type_${directionId}_${stop.stopId}_${selectedTimeType}`;
    const performance = globalData.performance?.[performanceKey];

    let value = 0;
    if (performance?.analytics?.punctuality?.punctuality_distribution?.percentages) {
      value = performance.analytics.punctuality.punctuality_distribution.percentages[selectedMetric] || 0;
    }

    return {
      ...stop,
      value,
      hasData: !!performance
    };
  });

  const maxValue = Math.max(...performanceData.map(d => d.value), 100);

  const getColor = (value: number, metric: string) => {
    const intensity = Math.min(value / maxValue, 1);
    switch (metric) {
      case 'on_time':
        return `rgba(34, 197, 94, ${0.3 + intensity * 0.7})`; // Green
      case 'too_late':
        return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`; // Red
      case 'too_early':
        return `rgba(245, 158, 11, ${0.3 + intensity * 0.7})`; // Yellow
      default:
        return `rgba(156, 163, 175, ${0.3 + intensity * 0.7})`; // Gray
    }
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

  const chartData = {
    labels: Object.values(sortedStops).map((stop) => stop.stopName),
    datasets: [
      {
        label: 'Punctuality Flow',
        data: performanceData.map((stop) => stop.value),
        backgroundColor: performanceData.map((stop) => getColor(stop.value, selectedMetric)),
        borderColor: performanceData.map((stop) => getColor(stop.value, selectedMetric)),
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: `Punctuality Flow - ${getMetricDisplayName(selectedMetric)} (${getTimeTypeDisplayName(selectedTimeType)})`,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Stop Names',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Punctuality',
        },
        beginAtZero: true,
      },
    },
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
              <div key={stop.stopId} className="flex flex-col items-center flex-1 group">
                {/* Bar */}
                <div className="relative w-full flex flex-col justify-end h-40">
                  <div
                    className="w-full rounded-t transition-all duration-200 group-hover:opacity-80 border border-gray-300"
                    style={{
                      height: `${(stop.value / maxValue) * 100}%`,
                      backgroundColor: getColor(stop.value, selectedMetric),
                      minHeight: stop.hasData ? '4px' : '2px',
                    }}
                  />

                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 whitespace-nowrap">
                    <div className="font-medium">{stop.stopName}</div>
                    <div className="text-center">
                      {stop.hasData ? `${stop.value.toFixed(1)}%` : 'No data'}
                    </div>
                    <div className="text-center text-gray-300">
                      Position {stop.position}
                    </div>
                  </div>
                </div>

                {/* Position label */}
                <div 
                  className="text-xs font-medium text-gray-600 mt-1"
                  style={{
                    fontWeight: stop.isRegulatory ? 'bolder' : 'normal' // Bold if regulatory stop
                  }}
                >
                  {stop.position}
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
              style={{ backgroundColor: getColor(maxValue, selectedMetric) }}
            />
            <span>{getMetricDisplayName(selectedMetric)} %</span>
          </div>
          <span>•</span>
          <span>Hover for details</span>
        </div>

        {/* Summary */}
        <div className="text-center text-sm text-gray-600">
          Showing {performanceData.filter((d) => d.hasData).length} of {performanceData.length} stops with data
        </div>

        {/* Performance Summary */}
        {performanceData.filter((d) => d.hasData).length > 0 && (
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-sm font-medium text-gray-700">Average</div>
                <div className="text-lg font-bold">{(performanceData.filter((d) => d.hasData).reduce((sum, d) => sum + d.value, 0) / performanceData.filter((d) => d.hasData).length).toFixed(1)}%</div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700">Best Stop</div>
                <div className="text-lg font-bold text-green-600">
                  {Math.min(...performanceData.filter((d) => d.hasData).map((d) => d.value)).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700">Worst Stop</div>
                <div className="text-lg font-bold text-red-600">
                  {Math.max(...performanceData.filter((d) => d.hasData).map((d) => d.value)).toFixed(1)}%
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
