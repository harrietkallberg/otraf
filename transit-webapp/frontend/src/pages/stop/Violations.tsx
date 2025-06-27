import React from 'react'

interface Props {
  data: any
}

const Violations: React.FC<Props> = ({ data }) => (
  <div>
    <h2 className="text-xl font-semibold mb-2">Stop Violations</h2>
    <pre>{JSON.stringify(data.violations_by_route, null, 2)}</pre>
  </div>
)

export default Violations