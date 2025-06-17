import React from 'react';
import { useParams } from 'react-router-dom';

export default function StopView() {
  const { stopId } = useParams();
  return (
    <div className="p-4">
      <h2 className="text-2xl font-semibold">Stop View: {stopId}</h2>
      {/* Load routes for this stop, violations, logs, etc */}
    </div>
  );
}