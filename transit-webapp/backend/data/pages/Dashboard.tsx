import React from 'react';

export default function Dashboard() {
  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold mb-4">Transit System Overview</h1>
      {/* You can fetch and display system stats here */}
      <p>Total Routes: [placeholder]</p>
      <p>Total Stops: [placeholder]</p>
      <p>Total Violations: [placeholder]</p>
    </div>
  );
}