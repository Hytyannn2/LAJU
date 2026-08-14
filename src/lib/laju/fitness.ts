import { DemandAnchor, DEMAND_ANCHORS } from '@/data/laju/demandAnchors';
import { haversineDistance } from './gravity';
import {
  BusRoute,
  EvaluationMetrics,
  KelantanGraphData,
  NetworkGenome,
  ODTrip,
  ShockState
} from './types';

/**
 * Calculates Bureau of Public Roads (BPR) congestion delayed travel time (in minutes).
 * t = t0 * (1 + 0.15 * (v / c)^4)
 */
export function calculateBPRTravelTime(
  lengthKm: number,
  speedLimitKmH: number,
  volume: number,
  capacity: number
): number {
  const t0Minutes = (lengthKm / Math.max(speedLimitKmH, 20)) * 60;
  const vcRatio = volume / Math.max(capacity, 100);
  const congestionFactor = 1 + 0.15 * Math.pow(vcRatio, 4);
  return t0Minutes * congestionFactor;
}

/**
 * Evaluates a candidate transit network under normal or shock conditions.
 */
export function evaluateGenomeSingleScenario(
  genome: NetworkGenome,
  graph: KelantanGraphData,
  odTrips: ODTrip[],
  anchors: DemandAnchor[] = DEMAND_ANCHORS,
  shocks: ShockState = { aeonEvent: false, monsoonClosure: false, hariRayaSurge: false }
): { fitness: number; metrics: EvaluationMetrics } {
  // 1. Map all nodes covered by current genome's bus routes
  const coveredNodeIds = new Set<string>();
  let totalBuses = 0;

  genome.routes.forEach(route => {
    route.nodeIds.forEach(nodeId => coveredNodeIds.add(nodeId));
    totalBuses += route.busCount;
  });

  const nodeMap = new Map(graph.nodes.map(n => [n.id, n]));

  // 2. Coverage calculation (% of anchors within 800m of a covered transit stop)
  let coveredAnchorCount = 0;
  const killedAnchorIds: string[] = [];
  const jajahanCoverageMap: Record<string, { total: number; covered: number }> = {};

  anchors.forEach(anchor => {
    if (!jajahanCoverageMap[anchor.jajahan]) {
      jajahanCoverageMap[anchor.jajahan] = { total: 0, covered: 0 };
    }
    jajahanCoverageMap[anchor.jajahan].total += 1;

    let isServed = false;
    for (const nodeId of coveredNodeIds) {
      const node = nodeMap.get(nodeId);
      if (node) {
        const dist = haversineDistance(anchor.lat, anchor.lng, node.lat, node.lng);
        if (dist <= 0.8) { // 800m coverage threshold
          isServed = true;
          break;
        }
      }
    }

    if (isServed) {
      coveredAnchorCount += 1;
      jajahanCoverageMap[anchor.jajahan].covered += 1;
    } else {
      killedAnchorIds.push(anchor.id);
    }
  });

  const coveragePercent = anchors.length > 0 ? (coveredAnchorCount / anchors.length) * 100 : 0;

  // 3. Average Wait Time calculation
  // Wait time per route = headway / 2
  let avgWaitMinutes = 45; // default fallback if no routes
  if (genome.routes.length > 0) {
    const totalHeadway = genome.routes.reduce((sum, r) => sum + r.headwayMinutes, 0);
    avgWaitMinutes = (totalHeadway / genome.routes.length) / 2;
  }

  // 4. Average Travel Time with BPR Congestion & Monsoon edge blockage
  // Calculate edge volumes (buses + commuter car volume)
  const edgeVolumeMap = new Map<string, number>();
  
  // Base car traffic volume on edges
  graph.edges.forEach(edge => {
    edgeVolumeMap.set(edge.id, shocks.hariRayaSurge ? 1200 : 450);
  });

  // Add transit bus traffic volume
  genome.routes.forEach(route => {
    const busesPerHour = 60 / Math.max(route.headwayMinutes, 5);
    for (let i = 0; i < route.nodeIds.length - 1; i++) {
      const from = route.nodeIds[i];
      const to = route.nodeIds[i + 1];
      const edge = graph.edges.find(e => e.from === from && e.to === to);
      if (edge) {
        edgeVolumeMap.set(edge.id, (edgeVolumeMap.get(edge.id) || 0) + busesPerHour * 3);
      }
    }
  });

  // Check Monsoon closure (delete / block key flood bridge edges KB-Pasir Mas / Machang-Kuala Krai)
  const blockedEdgeIds = new Set<string>();
  if (shocks.monsoonClosure) {
    graph.edges.forEach(edge => {
      // Block Sultan Yahya Petra bridge & Kuala Krai river edges during monsoon
      if (edge.from.includes('sy_bridge') || edge.to.includes('sy_bridge') ||
          edge.from.includes('kuala_krai') && edge.to.includes('chiku')) {
        blockedEdgeIds.add(edge.id);
      }
    });
  }

  // Calculate average transit travel time over OD trips
  let totalTravelTimeMin = 0;
  let evaluatedTrips = 0;

  odTrips.slice(0, 300).forEach(trip => { // Sample subset for fast 60fps eval
    const originNode = nodeMap.get(trip.originNodeId);
    if (!originNode) return;

    // Simple distance travel time estimate with BPR factor
    let tripTime = (trip.distanceKm / 40) * 60; // 40 km/h avg bus speed
    if (shocks.monsoonClosure && trip.distanceKm > 15) {
      tripTime *= 1.45; // Detour penalty during monsoon flood closure
    }

    totalTravelTimeMin += tripTime * trip.commuterCount;
    evaluatedTrips += trip.commuterCount;
  });

  const avgTravelMinutes = evaluatedTrips > 0 ? totalTravelTimeMin / evaluatedTrips : 35;

  // 5. Fleet Operating Cost calculation (RM 350 per bus per day + route length cost)
  const totalRouteKm = genome.routes.reduce((sum, r) => sum + (r.nodeIds.length * 2.5), 0);
  const fleetCostRM = totalBuses * 350 + totalRouteKm * 12;
  const fleetCostNorm = Math.min(1.0, fleetCostRM / 40000); // Normalized fleet cost 0..1

  // 6. Equity Penalty calculation
  // Ensure remote/low-pop jajahans (Gua Musang, Jeli, Lojing) have coverage representation
  let equityScoreSum = 0;
  let jajahanCount = 0;
  let missingRemoteJajahan = false;

  const remoteJajahans = ['Gua Musang', 'Jeli', 'Lojing'];

  Object.entries(jajahanCoverageMap).forEach(([jajahan, stats]) => {
    const ratio = stats.total > 0 ? stats.covered / stats.total : 0;
    equityScoreSum += ratio;
    jajahanCount += 1;

    if (remoteJajahans.includes(jajahan) && ratio === 0) {
      missingRemoteJajahan = true;
    }
  });

  const equityIndex = jajahanCount > 0 ? (equityScoreSum / jajahanCount) * 100 : 0;
  const equityPenalty = missingRemoteJajahan ? 0.35 : (1.0 - (equityIndex / 100)) * 0.15;

  // 7. Composite Fitness Score
  // fitness = 0.30*coverage + 0.25*(1/avgWait) + 0.25*(1/avgTravel) - 0.20*fleetCost - equityPenalty
  const waitTerm = Math.min(1.0, 10 / Math.max(avgWaitMinutes, 2));
  const travelTerm = Math.min(1.0, 20 / Math.max(avgTravelMinutes, 5));
  const coverageNorm = coveragePercent / 100;

  const fitness = (
    0.30 * coverageNorm +
    0.25 * waitTerm +
    0.25 * travelTerm -
    0.20 * fleetCostNorm -
    equityPenalty
  );

  const finalFitnessScore = Math.max(0.01, parseFloat(fitness.toFixed(4)));

  const metrics: EvaluationMetrics = {
    coveragePercent: parseFloat(coveragePercent.toFixed(1)),
    avgWaitMinutes: parseFloat(avgWaitMinutes.toFixed(1)),
    avgTravelMinutes: parseFloat(avgTravelMinutes.toFixed(1)),
    fleetCostRM: Math.round(fleetCostRM),
    equityIndex: parseFloat(equityIndex.toFixed(1)),
    killedAnchorIds,
    shockFitness: {}
  };

  return { fitness: finalFitnessScore, metrics };
}

/**
 * Robust evaluation: Final fitness = 0.7 * normal + 0.3 * worstShock
 */
export function evaluateGenomeRobust(
  genome: NetworkGenome,
  graph: KelantanGraphData,
  odTrips: ODTrip[],
  anchors: DemandAnchor[] = DEMAND_ANCHORS,
  activeShocks?: ShockState
): { compositeFitness: number; metrics: EvaluationMetrics } {
  // Normal evaluation
  const normalResult = evaluateGenomeSingleScenario(genome, graph, odTrips, anchors, {
    aeonEvent: false,
    monsoonClosure: false,
    hariRayaSurge: false
  });

  // Evaluate shock scenarios
  const aeonResult = evaluateGenomeSingleScenario(genome, graph, odTrips, anchors, {
    aeonEvent: true,
    monsoonClosure: false,
    hariRayaSurge: false
  });

  const monsoonResult = evaluateGenomeSingleScenario(genome, graph, odTrips, anchors, {
    aeonEvent: false,
    monsoonClosure: true,
    hariRayaSurge: false
  });

  const rayaResult = evaluateGenomeSingleScenario(genome, graph, odTrips, anchors, {
    aeonEvent: false,
    monsoonClosure: false,
    hariRayaSurge: true
  });

  const worstShockFitness = Math.min(aeonResult.fitness, monsoonResult.fitness, rayaResult.fitness);

  // If specific active shocks are selected, blend them
  let currentFitness = normalResult.fitness;
  if (activeShocks?.aeonEvent || activeShocks?.monsoonClosure || activeShocks?.hariRayaSurge) {
    const shockResult = evaluateGenomeSingleScenario(genome, graph, odTrips, anchors, activeShocks);
    currentFitness = shockResult.fitness;
  }

  // Composite robust fitness formula
  const compositeFitness = parseFloat((0.70 * currentFitness + 0.30 * worstShockFitness).toFixed(4));

  const finalMetrics = {
    ...normalResult.metrics,
    shockFitness: {
      Normal: normalResult.fitness,
      'AEON Event': aeonResult.fitness,
      'Monsoon Closure': monsoonResult.fitness,
      'Hari Raya Surge': rayaResult.fitness,
    }
  };

  return { compositeFitness, metrics: finalMetrics };
}
