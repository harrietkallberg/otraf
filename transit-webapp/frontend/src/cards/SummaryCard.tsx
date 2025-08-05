import React from 'react';
import { Summary } from '../shared/types'; // Import the correct interface

interface SummaryCardProps {
  title: string;
  data: Summary | null;  // Using the correct type for the data prop
}

const SummaryCard: React.FC<SummaryCardProps> = ({ title, data }) => {
  if (!data) {
    return <div>No data available</div>;
  }

  return (
    <div>
      <h3>{title}</h3>
      <div>
        <h4>Stop Topology:</h4>
        <p>Parent Stations: {data.stop_topology.labels_by_type.parent_station}</p>
        <p>Stop IDs: {data.stop_topology.labels_by_type.stop_id}</p>
        <p>Violations by Type: {JSON.stringify(data.stop_topology.violations_by_type)}</p>
      </div>

      <div>
        <h4>Direction Topology:</h4>
        <p>Direction IDs: {data.direction_topology.labels_by_type.direction_id}</p>
        <p>Stop IDs: {data.direction_topology.labels_by_type.stop_id}</p>
        <p>Violations by Type: {JSON.stringify(data.direction_topology.violations_by_type)}</p>
      </div>

      <div>
        <h4>Performance:</h4>
        <p>Available Performance Analytics: {data.performance.available_performace_analytics}</p>
      </div>

      {data.total_trip_instances !== undefined && (
      <div>
        <h4>Total Trip Instances:</h4>
        <p>{data.total_trip_instances}</p>
      </div>
    )}
    </div>
  );
};

export default SummaryCard;

