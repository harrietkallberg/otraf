// src/components/shared/LabelsViolationsDisplay.tsx
import React from 'react';

interface LabelsViolationsDisplayProps {
  labels: any[];
  violations: any[];
  size?: 'sm' | 'md' | 'lg';
}

export const LabelsViolationsDisplay: React.FC<LabelsViolationsDisplayProps> = ({ 
  labels, 
  violations, 
  size = 'md' 
}) => {
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';
  const padding = size === 'sm' ? 'p-2' : 'p-3';
  const spacing = size === 'sm' ? 'space-y-1' : 'space-y-2';

  const getSeverityColor = (severity: number) => {
    if (severity >= 5) return 'bg-red-100 text-red-800 border-red-200'
    if (severity >= 3) return 'bg-orange-100 text-orange-800 border-orange-200'
    return 'bg-yellow-100 text-yellow-800 border-yellow-200'
  };

  return (
    <div className={spacing}>
      {labels && labels.length > 0 && (
        <div>
          <h4 className={`${textSize} font-medium text-gray-700 mb-2`}>Labels ({labels.length})</h4>
          <div className={spacing}>
            {labels.map((label, index) => (
              <div key={index} className={`bg-blue-50 border border-blue-200 rounded-lg ${padding}`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {label.label_type}
                      </span>
                    </div>
                    <p className={`${textSize} text-gray-700`}>{label.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {violations && violations.length > 0 && (
        <div>
          <h4 className={`${textSize} font-medium text-gray-700 mb-2`}>Violations ({violations.length})</h4>
          <div className={spacing}>
            {violations.map((violation, index) => (
              <div key={index} className={`bg-red-50 border border-red-200 rounded-lg ${padding}`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        {violation.violation_type}
                      </span>
                      {violation.severity && (
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(violation.severity)}`}>
                          Severity {violation.severity}
                        </span>
                      )}
                    </div>
                    <p className={`${textSize} text-gray-700`}>{violation.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(!labels || labels.length === 0) && (!violations || violations.length === 0) && (
        <div className="text-center py-4 text-gray-500 text-xs">
          No labels or violations found
        </div>
      )}
    </div>
  );
};
