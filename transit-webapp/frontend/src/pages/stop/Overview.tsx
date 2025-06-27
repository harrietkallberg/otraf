import React from 'react'

interface Props {
  data: any
}

const Overview: React.FC<Props> = ({ data }) => (
  <div>
    <h2 className="text-xl font-semibold mb-2">Stop Details</h2>
    <p><strong>Name:</strong> {data.meta.stop_name}</p>
    <p><strong>ID:</strong> {data.meta.stop_id}</p>
    {/* add other metadata here */}
  </div>
)

export default Overview