// src/utils/dataLoader.ts

import type { RouteSummary, ViolationEntry, DelayHistogram, PunctualityEntry } from '../types';

export const loadJson = async <T = any>(path: string): Promise<T> => {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return await res.json();
};

export const loadRouteSummaries = (): Promise<RouteSummary[]> =>
  loadJson<RouteSummary[]>("/data/routes_summary.json");

export const loadStopViolations = (): Promise<ViolationEntry[]> =>
  loadJson<ViolationEntry[]>("/data/global_stop_violations.json");

export const loadDirectionViolations = (): Promise<ViolationEntry[]> =>
  loadJson<ViolationEntry[]>("/data/global_direction_violations.json");

export const loadPerformanceViolations = (): Promise<ViolationEntry[]> =>
  loadJson<ViolationEntry[]>("/data/global_performance_violations.json");

export const loadDelayHistogram = (routeId: string, directionId: string, timeType: string): Promise<DelayHistogram> =>
  loadJson<DelayHistogram>(`/data/routes/${routeId}/delay_histograms_${directionId}_${timeType}.json`);

export const loadPunctualityFlow = (routeId: string, directionId: string, timeType: string): Promise<PunctualityEntry[]> =>
  loadJson<PunctualityEntry[]>(`/data/routes/${routeId}/punctuality_flow_${directionId}_${timeType}.json`);
