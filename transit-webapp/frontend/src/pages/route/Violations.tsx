import React, { useEffect, useState } from 'react'

interface Props {
  nav: any
}

const Violations: React.FC<Props> = ({ nav }) => {
  const [violations, setViolations] = useState<any>(null)

  useEffect(() => {
    const dir = nav.directions[0].direction_id
    fetch(`/api/routes/${nav.route_id}/directions/${dir}/violations`)
      .then(r => r.json())
      .then(setViolations)
  }, [nav])

  if (!violations) return <div>Loading violations...</div>

  return (
    <div>
      <h2 className="text-xl font-semibold mb-2">Regulatory Violations</h2>
      <pre>{JSON.stringify(violations, null, 2)}</pre>
    </div>
  )
}

export default Violations