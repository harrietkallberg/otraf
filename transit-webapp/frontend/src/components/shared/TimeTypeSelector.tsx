// src/components/shared/TimeTypeSelector.tsx
import React from 'react';

interface TimeTypeSelectorProps {
  selectedTimeType: string;
  availableTimeTypes: string[];
  onTimeTypeChange: (timeType: string) => void;
  size?: 'sm' | 'md';
  label?: string;
}

export const TimeTypeSelector: React.FC<TimeTypeSelectorProps> = ({
  selectedTimeType,
  availableTimeTypes,
  onTimeTypeChange,
  size = 'md',
  label = 'Time Type:'
}) => {
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';
  const padding = size === 'sm' ? 'px-2 py-1' : 'px-3 py-1';
  
  return (
    <div className="flex items-center space-x-2">
      <label className={`${textSize} font-medium text-gray-700`}>{label}</label>
      <select
        value={selectedTimeType}
        onChange={(e) => onTimeTypeChange(e.target.value)}
        className={`${textSize} border border-gray-300 rounded-md ${padding} focus:outline-none focus:ring-1 focus:ring-blue-500`}
      >
        {availableTimeTypes.map((timeType: string) => (
          <option key={timeType} value={timeType}>
            {timeType.replace('_', ' ').toUpperCase()}
          </option>
        ))}
      </select>
    </div>
  );
};