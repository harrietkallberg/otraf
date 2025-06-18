// src/types.ts

export type TimeType = 'am_rush' | 'day' | 'pm_rush' | 'night' | 'weekend';

export interface RouteSummary {
  route_id: string;
  route_short_name: string;
  route_long_name: string;
  violation_counts: {
    stop_topology: number;
    direction_topology: number;
    regulatory: number;
  };
}

export interface ViolationEntry {
  route_id: string;
  direction_id?: string;
  stop_id?: string;
  stop_name?: string;
  time_type?: TimeType;
  violation_type: string;
  details: any;
}

export interface DelayHistogram {
  bins: number[];
  total_delays: number[];
  incremental_delays: number[];
}

export interface PunctualityDistribution {
  too_early: number;
  on_time: number;
  too_late: number;
}

export interface PunctualityEntry {
  stop_id: string;
  stop_name: string;
  direction_id: string;
  time_type: TimeType;
  punctuality: PunctualityDistribution;
}