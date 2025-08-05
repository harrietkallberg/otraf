// src/hooks/useProgressiveSearch.ts
import { useState, useEffect, useMemo } from 'react';

export interface SearchFilters {
  [key: string]: string;
}

export interface FilterValidationRule<T> {
  key: string;
  validateItem: (item: T, filters: SearchFilters) => boolean;
  extractOptions: (item: T) => string | string[] | number | number[];
  dependencies?: string[]; // Which other filters this depends on
}

export interface UseProgressiveSearchProps<T> {
  data: T[];
  initialFilters?: SearchFilters;
  filterRules: FilterValidationRule<T>[];
  hasValidData?: (item: T) => boolean;
}

export interface UseProgressiveSearchResult<T> {
  filters: SearchFilters;
  setFilter: (key: string, value: string) => void;
  filteredData: T[];
  availableOptions: { [key: string]: Set<string> };
  clearFilters: () => void;
  isFilterValid: (key: string, value: string) => boolean;
}

export function useProgressiveSearch<T>({
  data,
  initialFilters = {},
  filterRules,
  hasValidData = () => true
}: UseProgressiveSearchProps<T>): UseProgressiveSearchResult<T> {
  
  const [filters, setFilters] = useState<SearchFilters>(initialFilters);

  // Helper function to update a single filter
  const setFilter = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  // Smart filter clearing - only clear if current selection becomes invalid
  useEffect(() => {
    if (!data.length) return;

    const validOptions: { [key: string]: Set<string> } = {};
    
    // Initialize all option sets
    filterRules.forEach(rule => {
      validOptions[rule.key] = new Set<string>();
    });

    // Collect valid options for each filter based on current selections
    data.forEach(item => {
      if (!hasValidData(item)) return;

      filterRules.forEach(rule => {
        // Check if this item matches all OTHER filters (not including current filter)
        const otherFilters = Object.fromEntries(
          Object.entries(filters).filter(([k]) => k !== rule.key)
        );
        
        const matchesOtherFilters = filterRules
          .filter(r => r.key !== rule.key)
          .every(r => r.validateItem(item, otherFilters));

        if (matchesOtherFilters) {
          const options = rule.extractOptions(item);
          const optionsArray = Array.isArray(options) ? options : [String(options)];
          optionsArray.forEach(option => validOptions[rule.key].add(String(option)));
        }
      });
    });

    // Clear invalid selections cascading from left to right
    const newFilters = { ...filters };
    let hasChanges = false;

    for (const rule of filterRules) {
      const currentValue = filters[rule.key];
      if (currentValue && !validOptions[rule.key].has(currentValue)) {
        newFilters[rule.key] = '';
        hasChanges = true;
        
        // Clear dependent filters
        if (rule.dependencies) {
          rule.dependencies.forEach(depKey => {
            if (newFilters[depKey]) {
              newFilters[depKey] = '';
              hasChanges = true;
            }
          });
        }
      }
    }

    if (hasChanges) {
      setFilters(newFilters);
    }
  }, [data, filters, filterRules, hasValidData]);

  // Progressive filtering and results calculation
  const { availableOptions, filteredData } = useMemo(() => {
    if (!data.length) {
      const emptyOptions: { [key: string]: Set<string> } = {};
      filterRules.forEach(rule => {
        emptyOptions[rule.key] = new Set<string>();
      });
      return { availableOptions: emptyOptions, filteredData: [] };
    }

    const options: { [key: string]: Set<string> } = {};
    const filtered: T[] = [];

    // Initialize all option sets
    filterRules.forEach(rule => {
      options[rule.key] = new Set<string>();
    });

    data.forEach(item => {
      if (!hasValidData(item)) return;

      // Check if item matches all current filters
      const matchesAllFilters = filterRules.every(rule => 
        rule.validateItem(item, filters)
      );

      // If matches all current filters, include in results
      if (matchesAllFilters) {
        filtered.push(item);
      }

      // Collect available options based on partial matches (progressive filtering)
      filterRules.forEach(rule => {
        // Check if this item matches all OTHER filters (not including current filter)
        const otherFilters = Object.fromEntries(
          Object.entries(filters).filter(([k]) => k !== rule.key)
        );
        
        const matchesOtherFilters = filterRules
          .filter(r => r.key !== rule.key)
          .every(r => r.validateItem(item, otherFilters));

        if (matchesOtherFilters) {
          const itemOptions = rule.extractOptions(item);
          const optionsArray = Array.isArray(itemOptions) ? itemOptions : [String(itemOptions)];
          optionsArray.forEach(option => options[rule.key].add(String(option)));
        }
      });
    });

    return { availableOptions: options, filteredData: filtered };
  }, [data, filters, filterRules, hasValidData]);

  const clearFilters = () => {
    setFilters({});
  };

  const isFilterValid = (key: string, value: string) => {
    return !value || availableOptions[key]?.has(value) || false;
  };

  return {
    filters,
    setFilter,
    filteredData,
    availableOptions,
    clearFilters,
    isFilterValid
  };
}