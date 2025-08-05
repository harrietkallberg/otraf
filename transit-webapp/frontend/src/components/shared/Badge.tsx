
// src/components/shared/Badge.tsx
import React from 'react';

interface BadgeProps {
  count: number;
  type: 'labels' | 'violations' | 'analytics' | 'regulatory';
  size?: 'sm' | 'md';
  customText?: string;
}

export const Badge: React.FC<BadgeProps> = ({ count, type, size = 'md', customText }) => {
  const getColors = () => {
    switch (type) {
      case 'labels': return 'bg-blue-100 text-blue-800';
      case 'violations': return 'bg-red-100 text-red-800';
      case 'analytics': 
        if (!customText) return 'bg-amber-100 text-amber-800';
        
        // Extract percentage from text like "XX:75%" or "75%"
        const percentMatch = customText.match(/(\d+(?:\.\d+)?)%/);
        if (percentMatch) {
          const percentage = parseFloat(percentMatch[1]);
          if (percentage < 50) {
            return 'bg-red-100 text-red-800';
          }
          if (percentage < 80) {
            return 'bg-amber-100 text-amber-800';
          }
        }
        
        return 'bg-emerald-100 text-emerald-800';
      case 'regulatory': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const textSize = size === 'sm' ? 'text-xs' : 'text-xs';
  
  return (
    <span className={`inline-flex items-center px-2 py-1 rounded-full ${textSize} font-medium ${getColors()}`}>
      {customText || `${count} ${type}`}
    </span>
  );
};
