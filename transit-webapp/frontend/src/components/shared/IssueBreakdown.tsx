// Issue Breakdown component that works with your actual data - all 5 severity levels
interface IssueBreakdownProps {
  labels: any[];
  violations: any[];
  performanceData?: any;
}

export const IssueBreakdown: React.FC<IssueBreakdownProps> = ({ 
  labels, 
  violations, 
  performanceData 
}) => {
  // Calculate issue metrics by severity level (1-5)
  const severity5Violations = violations.filter(v => v.severity === 5).length;
  const severity4Violations = violations.filter(v => v.severity === 4).length;
  const severity3Violations = violations.filter(v => v.severity === 3).length;
  const severity2Violations = violations.filter(v => v.severity === 2).length;
  const severity1Violations = violations.filter(v => v.severity === 1).length;
  
  // Count regulatory-related items
  const regulatoryLabels = labels.filter(l => 
    l.label_type?.includes('regulatory') || 
    l.description?.toLowerCase().includes('regulatory')
  ).length;
  
  const regulatoryViolations = violations.filter(v => 
    v.violation_type?.includes('regulatory') || 
    v.description?.toLowerCase().includes('regulatory')
  ).length;
  
  const totalRegulatory = regulatoryLabels + regulatoryViolations;
  
  // Performance issues
  const hasPerformanceIssues = performanceData?.analytics?.punctuality?.punctuality_distribution?.percentages?.on_time < 80;
  const hasHighDelays = performanceData?.analytics?.punctuality?.basic_statistics?.mean_delay > 120;
  
  const totalIssues = violations.length + (hasPerformanceIssues ? 1 : 0) + (hasHighDelays ? 1 : 0);

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h4 className="text-sm font-medium text-gray-700 mb-3">Issue Summary</h4>
      <div className="space-y-3">
        
        {/* Severity 5 - Critical */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-600"></div>
            <span className="text-sm text-gray-600">Critical (Severity 5)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity5Violations > 0 ? 'text-red-600' : 'text-gray-400'
          }`}>
            {severity5Violations}
          </span>
        </div>

        {/* Severity 4 - High */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-400"></div>
            <span className="text-sm text-gray-600">High (Severity 4)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity4Violations > 0 ? 'text-red-500' : 'text-gray-400'
          }`}>
            {severity4Violations}
          </span>
        </div>

        {/* Severity 3 - Medium */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-orange-500"></div>
            <span className="text-sm text-gray-600">Medium (Severity 3)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity3Violations > 0 ? 'text-orange-600' : 'text-gray-400'
          }`}>
            {severity3Violations}
          </span>
        </div>

        {/* Severity 2 - Low */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span className="text-sm text-gray-600">Low (Severity 2)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity2Violations > 0 ? 'text-yellow-600' : 'text-gray-400'
          }`}>
            {severity2Violations}
          </span>
        </div>

        {/* Severity 1 - Minimal */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span className="text-sm text-gray-600">Minimal (Severity 1)</span>
          </div>
          <span className={`text-sm font-medium ${
            severity1Violations > 0 ? 'text-blue-600' : 'text-gray-400'
          }`}>
            {severity1Violations}
          </span>
        </div>

        {/* Regulatory Issues */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-amber-500"></div>
            <span className="text-sm text-gray-600">Regulatory Issues</span>
          </div>
          <span className={`text-sm font-medium ${
            totalRegulatory > 0 ? 'text-amber-600' : 'text-gray-400'
          }`}>
            {totalRegulatory}
          </span>
        </div>

        {/* Labels Applied */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-sm text-gray-600">Labels Applied</span>
          </div>
          <span className={`text-sm font-medium ${
            labels.length > 0 ? 'text-green-600' : 'text-gray-400'
          }`}>
            {labels.length}
          </span>
        </div>

        {/* No Issues Found */}
        {totalIssues === 0 && labels.length === 0 && (
          <div className="text-center py-2">
            <span className="text-sm text-green-600 font-medium">✓ No Issues Found</span>
          </div>
        )}
      </div>
    </div>
  );
};
