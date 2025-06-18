
// src/pages/StopDetail.tsx
import React from 'react';
import { useParams } from 'react-router-dom';

const StopDetail: React.FC = () => {
  const { stopId } = useParams();
  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Stop Details: {stopId}</h1>
      <p>Stop-centric view and violations will go here.</p>
    </div>
  );
};

export default StopDetail;