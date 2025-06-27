import React from 'react'

interface Props {
  data: any
}

const Performance: React.FC<Props> = ({ data }) => (
  <div>
    <h2 className="text-xl font-semibold mb-2">Stop Performance</h2>
    {/* render your histograms by time-type here, e.g. */}
    <pre>{JSON.stringify(data.meta, null, 2)}</pre>
  </div>
)

export default Performance