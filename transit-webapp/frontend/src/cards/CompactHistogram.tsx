import React, { useState } from 'react';

interface CompactHistogramProps {
  performanceData: any;  // Performance data that contains the histograms
}


// Compact Histogram Component
const CompactHistogram: React.FC<{
  performanceData: any
}> = ({ performanceData }) => {
  const [showTotalDelay, setShowTotalDelay] = useState(true)
  const [showIncrementalDelay, setShowIncrementalDelay] = useState(false)
  
  const totalDelayHist = performanceData.analytics.total_delay_histogram
  const incrementalDelayHist = performanceData.analytics.incremental_delay_histogram
  
  if (!totalDelayHist && !incrementalDelayHist) return null

  // Generate simplified labels for key points
  const generateSimplifiedLabels = (stepStart: number, stepSize: number, proportions: number[]) => {
    const labels = []
    const keyIndices = []
    
    for (let i = 0; i < proportions.length; i++) {
      const binStart = stepStart + (i * stepSize)
      // Mark key points: -60, 0, 60, 120, etc.
      if (binStart % 60 === 0 || i === 0 || i === proportions.length - 1) {
        keyIndices.push(i)
        labels.push(binStart >= 0 ? `${binStart}s` : `${binStart}s`)
      } else {
        labels.push('')
      }
    }
    return { labels, keyIndices }
  }

  const { labels, keyIndices } = totalDelayHist ? generateSimplifiedLabels(
    totalDelayHist.step_start,
    totalDelayHist.step_size,
    totalDelayHist.proportions
  ) : { labels: [], keyIndices: [] }

  // Find max proportion for scaling
  const maxProportion = Math.max(
    ...(showTotalDelay && totalDelayHist ? totalDelayHist.proportions : [0]),
    ...(showIncrementalDelay && incrementalDelayHist ? incrementalDelayHist.proportions : [0])
  )

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h5 className="text-xs font-medium text-gray-600">DELAY HISTOGRAM</h5>
        <div className="flex items-center space-x-2">
          {totalDelayHist && (
            <label className="flex items-center space-x-1 text-xs cursor-pointer">
              <input
                type="checkbox"
                checked={showTotalDelay}
                onChange={(e) => setShowTotalDelay(e.target.checked)}
                className="w-3 h-3 rounded"
              />
              <div className="w-2 h-2 bg-blue-500 rounded"></div>
            </label>
          )}
          {incrementalDelayHist && (
            <label className="flex items-center space-x-1 text-xs cursor-pointer">
              <input
                type="checkbox"
                checked={showIncrementalDelay}
                onChange={(e) => setShowIncrementalDelay(e.target.checked)}
                className="w-3 h-3 rounded"
              />
              <div className="w-2 h-2 bg-orange-500 rounded"></div>
            </label>
          )}
        </div>
      </div>
      
      <div className="relative">
        {/* Histogram bars */}
        <div className="grid gap-px items-end h-16 mb-2" style={{ gridTemplateColumns: `repeat(${labels.length}, 1fr)` }}>
          {labels.map((label, index) => (
            <div key={index} className="flex flex-col items-center relative">
              <div className="w-full flex justify-center items-end h-14">
                {/* Total Delay Bar */}
                {showTotalDelay && totalDelayHist && (
                  <div 
                    className="bg-blue-500 transition-all duration-300 hover:bg-blue-600 rounded-t"
                    style={{ 
                      height: `${(totalDelayHist.proportions[index] / maxProportion) * 100}%`,
                      width: showIncrementalDelay ? '45%' : '80%',
                      minHeight: totalDelayHist.proportions[index] > 0 ? '1px' : '0px',
                      marginRight: showIncrementalDelay ? '1px' : '0'
                    }}
                    title={`Total: ${(totalDelayHist.proportions[index] * 100).toFixed(1)}%`}
                  />
                )}
                
                {/* Incremental Delay Bar */}
                {showIncrementalDelay && incrementalDelayHist && (
                  <div 
                    className="bg-orange-500 transition-all duration-300 hover:bg-orange-600 rounded-t"
                    style={{ 
                      height: `${(incrementalDelayHist.proportions[index] / maxProportion) * 100}%`,
                      width: showTotalDelay ? '45%' : '80%',
                      minHeight: incrementalDelayHist.proportions[index] > 0 ? '1px' : '0px',
                      marginLeft: showTotalDelay ? '1px' : '0'
                    }}
                    title={`Incremental: ${(incrementalDelayHist.proportions[index] * 100).toFixed(1)}%`}
                  />
                )}
              </div>
            </div>
          ))}
        </div>
        
        {/* X-axis labels */}
        <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${labels.length}, 1fr)` }}>
          {labels.map((label, index) => (
            <div key={index} className="text-center">
              {keyIndices.includes(index) && (
                <span className="text-xs text-gray-500">{label}</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default CompactHistogram;
