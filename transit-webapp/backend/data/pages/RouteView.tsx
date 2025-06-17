import React from 'react';
import { useParams } from 'react-router-dom';

export default function RouteView() {
  const { routeId } = useParams();
  return (
    <div className="p-4">
      <h2 className="text-2xl font-semibold">Route View: {routeId}</h2>
      {/* Load direction topology, stops per direction, etc */}
    </div>
  );
}