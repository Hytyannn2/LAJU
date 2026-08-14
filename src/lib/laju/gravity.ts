import { KELANTAN_POPULATION } from '@/data/laju/kelantanPopulation';
import { DEMAND_ANCHORS, DemandAnchor } from '@/data/laju/demandAnchors';
import { GraphNode, ODTrip } from './types';

export function haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Earth radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Generates an Origin-Destination (OD) matrix of synthetic commuters.
 * Sample ~50k commuters scaled proportionally by:
 * trips_ij ∝ (pop_i × attractorWeight_j) / dist²
 */
export function generateODMatrix(
  nodes: GraphNode[],
  anchors: DemandAnchor[] = DEMAND_ANCHORS,
  totalCommuters = 50000,
  aeonBoost = false
): ODTrip[] {
  const rawTrips: { originNodeId: string; destAnchorId: string; rawScore: number; distKm: number }[] = [];
  let totalRawScore = 0;

  // Compute raw gravity scores for node -> anchor pairs
  nodes.forEach(node => {
    const nodeJajahanPop = KELANTAN_POPULATION[node.jajahan] || 50000;
    
    anchors.forEach(anchor => {
      const dist = haversineDistance(node.lat, node.lng, anchor.lat, anchor.lng);
      // Avoid division by near zero, set minimum distance threshold at 0.5km
      const safeDist = Math.max(dist, 0.5);

      // Apply AEON KB multiplier if shock mode active
      let weight = anchor.weight;
      if (aeonBoost && anchor.id === 'kb_02') {
        weight *= 8;
      }

      // Gravity Law: trips ∝ (Pop_i * Weight_j) / dist²
      const rawScore = (nodeJajahanPop * weight) / Math.pow(safeDist, 2);
      
      // Filter out negligible distant trips (>80km) to keep matrix performant
      if (dist <= 85) {
        rawTrips.push({
          originNodeId: node.id,
          destAnchorId: anchor.id,
          rawScore,
          distKm: dist
        });
        totalRawScore += rawScore;
      }
    });
  });

  if (totalRawScore === 0) return [];

  // Scale raw gravity scores to sample exactly ~50,000 synthetic commuters
  const odTrips: ODTrip[] = rawTrips.map(trip => {
    const commuterCount = Math.round((trip.rawScore / totalRawScore) * totalCommuters);
    return {
      originNodeId: trip.originNodeId,
      destAnchorId: trip.destAnchorId,
      commuterCount: Math.max(1, commuterCount),
      distanceKm: trip.distKm
    };
  }).filter(t => t.commuterCount > 0);

  return odTrips;
}
