// Summary interface for route and stop data (shared structure)
export interface Summary {
  stop_topology: {
    labels_by_type: {
      parent_station: number;
      stop_id: number;
    };
    violations_by_type: {
      parent_station: number;
      stop_id: number;
    };
  };
  direction_topology: {
    labels_by_type: {
      direction_id: number;
      stop_id: number;
    };
    violations_by_type: {
      direction_id: number;
      stop_id: number;
    };
  };

  regulatory_stops: {
      regulatory_stop_ids: number;
  };

  performance: {
    available_performace_analytics: number;
  };
  
  total_trip_instances?: number;
}

// PerformanceSummary type for performance metrics
export interface PerformanceSummary {
  overall_too_early_rate: number;
  overall_on_time_rate: number;
  overall_too_late_rate: number;
  average_departure_delay: number;
  canonical_share: number;
}
export interface StopData {
    parent_station: string;
    stop_name: string;
    stop_ids: string[];
    on_routes: string[];
    stop_summary: Summary  | null; 
    performance_summary: PerformanceSummary | null;  // Performance data for the route
    routes: { [routeId: string]: RouteData } | null;  // Summary of the rout
}
export interface RouteData {
    route_id: string;
    route_long_name: string;
    route_short_name: string;
    on_stops: string[];
    route_summary: Summary | null;  // Summary of the route
    performance_summary: PerformanceSummary | null;  // Performance data for the route
    directions: { [directionId: string]: DirectionData } | null;
}

// DirectionSummary includes direction-specific data for the route
export interface DirectionData {
  direction_id: string;
  direction_summary: Summary;  // Summary of the direction
  stop_ids_in_direction: { [position: string]: StopIdData } | null;  // Stops in this direction
}

// Full structure for a stop ID (including labels, violations, performance, and summary)
export interface StopIdData {
  stop_id: string;  // The stop ID
  stop_name: string;  // The name of the stop
  parent_station: string;  // The parent station's ID
  position?: number;

  // Keys related to direction and parent station topology
  direction_id_label_key: string[];  // Direction topology label keys
  direction_id_violation_key: string[];  // Direction topology violation keys
  parent_station_label_key: string[];  // Parent station label keys
  parent_station_violation_key: string[];  // Parent station violation keys

  // Keys related to stop ID labels and performance
  stop_id_label_keys: string[];  // Stop ID label keys (includes both stop and direction topology)
  stop_id_violation_keys: string[];  // Violation keys for the stop ID
  stop_id_performance_keys: string[];  // Performance keys for the stop (categorized by time type)

  // Summary for the stop ID (including stop topology, direction topology, regulatory stops, and performance)
  stop_id_summary: Summary;  // Detailed summary for the stop ID
}

export interface ResultCombination {
  routeId?: string
  directionId?: string
  stop_id?: string
  time_type?: string
  stopIdData: StopIdData
}

export interface StopDetailsData {
  labels: any[];
  violations: any[];
  performanceData: any;
}

export interface BadgeData {
  count: number;
  type: 'labels' | 'violations' | 'analytics' | 'regulatory';
  variant?: string;
}


export interface RouteDirectionTileProps {
  routeId: string;
  directionId: string;
  direction: any; // DirectionData from the route context
  parentStation: string;
  globalData: any;
}

export interface StopIdDataWithPosition extends StopIdData {
  position: number;
}

export type ComponentSize = 'sm' | 'md' | 'lg';