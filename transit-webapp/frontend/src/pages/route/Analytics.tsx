import React from 'react'

interface Props {
  nav: any
}

const Analytics: React.FC<Props> = ({ nav }) => (
  <div>
    <h2 className="text-xl font-semibold mb-2">Route Analytics</h2>
    {/* pull histograms & travel times from nav or other endpoints */}
    <p>Histogram and travel-time charts go here.</p>
  </div>
)

export default Analytics