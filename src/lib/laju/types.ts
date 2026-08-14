import { DemandAnchor } from '@/data/laju/demandAnchors';

export interface GraphNode {
  id: string;
  name: string;
  lat: number;
  lng: number;
  jajahan: string;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  lengthKm: number;
  oneway: boolean;
  capacity: number;
  speedLimitKmH: number;
}

export interface KelantanGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ODTrip {
  originNodeId: string;
  destAnchorId: string;
  commuterCount: number; // Volume of commuters taking this trip per day
  distanceKm: number;
}

export interface BusRoute {
  id: string;
  name: string;
  color: string;
  nodeIds: string[]; // Ordered list of node IDs forming the bus route path
  headwayMinutes: number; // Interval between buses (e.g., 10, 15, 30 mins)
  busCount: number; // Number of buses operating on this route
}

export interface NetworkGenome {
  id: string;
  routes: BusRoute[];
  fitnessScore: number;
}

export interface EvaluationMetrics {
  coveragePercent: number; // % of demand anchors within 800m of a transit route
  avgWaitMinutes: number; // Avg passenger wait time
  avgTravelMinutes: number; // Avg passenger travel time in transit system
  fleetCostRM: number; // Daily fleet operating cost
  equityIndex: number; // 0 to 100 equity score across jajahans
  killedAnchorIds: string[]; // Anchors with zero coverage -> recommend DRT
  shockFitness: Record<string, number>; // Fitness breakdown per shock mode
}

export type DayOfWeek = 'Sunday' | 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday' | 'Saturday';

export interface ShockState {
  aeonEvent: boolean; // 8x demand multiplier at AEON Mall KB
  monsoonClosure: boolean; // Bridge/road edge blockage
  hariRayaSurge: boolean; // 3x demand state-wide
}
