
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
      case 'analytics': return 'bg-green-100 text-green-800';
      case 'regulatory': return 'bg-amber-100 text-amber-800';
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
