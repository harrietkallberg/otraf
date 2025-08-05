// src/components/shared/ProgressiveSearchFilters.tsx
import React from 'react';

export interface FilterOption {
  value: string | number;
  label: string;
  disabled?: boolean;
}

export interface FilterConfig {
  key: string;
  label: string;
  placeholder: string;
  options: FilterOption[];
  value: string | number;
  onChange: (value: string) => void;
  className?: string;
}

interface ProgressiveSearchFiltersProps {
  title?: string;
  subtitle?: string;
  filters: FilterConfig[];
  className?: string;
}

export const ProgressiveSearchFilters: React.FC<ProgressiveSearchFiltersProps> = ({
  title = "Filter Results",
  subtitle,
  filters,
  className = ""
}) => {
  return (
    <div className={`bg-white shadow-sm rounded-lg border border-gray-200 p-6 ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900">{title}</h3>
        {subtitle && (
          <p className="text-sm text-gray-600 mt-1">{subtitle}</p>
        )}
      </div>
      
      <div className={`grid gap-4 ${filters.length <= 4 ? 'grid-cols-1 md:grid-cols-4' : 'grid-cols-1 md:grid-cols-5'}`}>
        {filters.map((filter) => (
          <div key={filter.key} className={filter.className}>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {filter.label}
            </label>
            <select
              value={filter.value}
              onChange={(e) => filter.onChange(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">{filter.placeholder}</option>
              {filter.options.map((option) => (
                <option 
                  key={option.value} 
                  value={option.value}
                  disabled={option.disabled}
                >
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProgressiveSearchFilters;