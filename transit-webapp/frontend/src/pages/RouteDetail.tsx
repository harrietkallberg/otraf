// src/pages/RouteDetail.tsx
import React from 'react';
import { useParams } from 'react-router-dom';

const RouteDetail: React.FC = () => {
  const { routeId } = useParams();
  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Route Details: {routeId}</h1>
      <p>Route-centric details and visualizations will go here.</p>
    </div>
  );
};

export default RouteDetail;