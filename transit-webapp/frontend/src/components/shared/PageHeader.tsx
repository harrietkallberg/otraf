// src/components/shared/PageHeader.tsx
import React, { useState } from 'react';

interface PageHeaderProps {
  title: string;
  helpText?: string;
  subtitle?: string;
  className?: string;
}

export const PageHeader: React.FC<PageHeaderProps> = ({ 
  title, 
  helpText, 
  subtitle,
  className = "" 
}) => {
  const [showHelp, setShowHelp] = useState(false);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header Section with Help */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <h1 className="text-3xl font-bold">{title}</h1>
          {helpText && (
            <button
              onClick={() => setShowHelp(!showHelp)}
              className="w-7 h-7 bg-blue-100 hover:bg-blue-200 rounded-full flex items-center justify-center transition-colors duration-200"
              title="Help"
            >
              <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
          )}
        </div>
        {subtitle && (
          <div className="text-sm text-gray-500">
            {subtitle}
          </div>
        )}
      </div>

      {/* Help Section */}
      {helpText && showHelp && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <svg className="w-5 h-5 text-blue-600 mt-0.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">{title} Help</h3>
              <p className="text-sm text-blue-800 leading-relaxed">
                {helpText}
              </p>
              <button
                onClick={() => setShowHelp(false)}
                className="mt-4 text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                Got it, hide help
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};