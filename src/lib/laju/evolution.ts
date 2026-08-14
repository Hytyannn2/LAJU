import { DEMAND_ANCHORS, DemandAnchor } from '@/data/laju/demandAnchors';
import { evaluateGenomeRobust } from './fitness';
import { generateODMatrix } from './gravity';
import {
  BusRoute,
  EvaluationMetrics,
  KelantanGraphData,
  NetworkGenome,
  ODTrip,
  ShockState
} from './types';

const ROUTE_COLORS = [
  '#ef4444', // Red (Express KB-Gua Musang)
  '#3b82f6', // Blue (KB-Pasir Mas-Rantau Panjang)
  '#10b981', // Green (KB-Tumpat)
  '#f59e0b', // Amber (KB-Bachok)
  '#8b5cf6', // Purple (KB-Pasir Puteh)
  '#ec4899', // Pink (KB-Machang)
  '#06b6d4', // Cyan (Machang-Tanah Merah-Jeli)
  '#84cc16', // Lime (Kuala Krai-Dabong)
  '#f97316', // Orange (Tok Bali Port Express)
];

/**
 * Creates an initial random candidate bus route along connected graph edges.
 */
export function createRandomRoute(id: string, name: string, color: string, graph: KelantanGraphData): BusRoute {
  const nodeCount = graph.nodes.length;
  const startIdx = Math.floor(Math.random() * nodeCount);
  const startNode = graph.nodes[startIdx];

  const pathNodeIds = [startNode.id];
  let currNodeId = startNode.id;

  // Build random connected path of 4 to 12 stops
  const targetLength = Math.floor(Math.random() * 8) + 4;
  for (let i = 0; i < targetLength; i++) {
    const outgoingEdges = graph.edges.filter(e => e.from === currNodeId);
    if (outgoingEdges.length === 0) break;
    const nextEdge = outgoingEdges[Math.floor(Math.random() * outgoingEdges.length)];
    if (pathNodeIds.includes(nextEdge.to)) continue; // Avoid simple loops
    pathNodeIds.push(nextEdge.to);
    currNodeId = nextEdge.to;
  }

  const busCount = Math.floor(Math.random() * 6) + 2;
  const headway = Math.floor(60 / busCount);

  return {
    id,
    name,
    color,
    nodeIds: pathNodeIds,
    headwayMinutes: headway,
    busCount
  };
}

/**
 * Creates a predefined seed route network (e.g. KB-Pasir Mas, KB-Tumpat, KB-Gua Musang corridors).
 */
export function createInitialSeedGenome(id: string, graph: KelantanGraphData): NetworkGenome {
  // Find key nodes for seeded corridors
  const kbCenter = graph.nodes.find(n => n.id.includes('kb_center')) || graph.nodes[0];
  const pmTown = graph.nodes.find(n => n.id.includes('pm_pasir_mas')) || graph.nodes[10];
  const tpTown = graph.nodes.find(n => n.id.includes('tp_tumpat')) || graph.nodes[20];
  const mcTown = graph.nodes.find(n => n.id.includes('mc_machang')) || graph.nodes[30];
  const kkTown = graph.nodes.find(n => n.id.includes('kk_kuala_krai')) || graph.nodes[40];
  const gmTown = graph.nodes.find(n => n.id.includes('gm_gua_musang')) || graph.nodes[50];
  const jlTown = graph.nodes.find(n => n.id.includes('jl_jeli')) || graph.nodes[60];

  const route1: BusRoute = {
    id: 'r_kb_pm',
    name: 'Corridor 1: KB - Pasir Mas - Rantau Panjang',
    color: ROUTE_COLORS[1],
    nodeIds: [kbCenter.id, 'kb_sy_bridge', 'kb_salor', 'pm_tendong', pmTown.id, 'pm_lati', 'pm_rantau_panjang'].filter(id => graph.nodes.some(n => n.id === id)),
    headwayMinutes: 15,
    busCount: 6
  };

  const route2: BusRoute = {
    id: 'r_kb_tp',
    name: 'Corridor 2: KB - Wakaf Bharu - Tumpat',
    color: ROUTE_COLORS[2],
    nodeIds: [kbCenter.id, 'tp_wakaf_bharu', 'tp_chabang_empat', tpTown.id, 'tp_pengkalan_kubor'].filter(id => graph.nodes.some(n => n.id === id)),
    headwayMinutes: 20,
    busCount: 4
  };

  const route3: BusRoute = {
    id: 'r_kb_gm',
    name: 'Corridor 3: KB - Machang - Kuala Krai - Gua Musang',
    color: ROUTE_COLORS[0],
    nodeIds: [kbCenter.id, 'kb_wakaf_che_yeh', 'kb_ketereh', 'kb_kok_lanas', mcTown.id, kkTown.id, 'gm_chiku', gmTown.id].filter(id => graph.nodes.some(n => n.id === id)),
    headwayMinutes: 30,
    busCount: 8
  };

  const route4: BusRoute = {
    id: 'r_kb_jl',
    name: 'Corridor 4: KB - Machang - Tanah Merah - Jeli',
    color: ROUTE_COLORS[6],
    nodeIds: [kbCenter.id, mcTown.id, 'tm_tanah_merah_town', 'tm_batugajah', jlTown.id, 'jl_umk_jeli'].filter(id => graph.nodes.some(n => n.id === id)),
    headwayMinutes: 30,
    busCount: 5
  };

  return {
    id,
    routes: [route1, route2, route3, route4],
    fitnessScore: 0
  };
}

/**
 * Initializes a population of candidate genomes.
 */
export function initializePopulation(
  popSize = 60,
  graph: KelantanGraphData
): NetworkGenome[] {
  const population: NetworkGenome[] = [];

  // Seed first 5 genomes with structured corridor networks
  for (let i = 0; i < 5; i++) {
    population.push(createInitialSeedGenome(`gen0_seed_${i}`, graph));
  }

  // Generate remaining random candidate genomes
  for (let i = 5; i < popSize; i++) {
    const routeCount = Math.floor(Math.random() * 4) + 3; // 3 to 6 routes per network
    const routes: BusRoute[] = [];
    for (let r = 0; r < routeCount; r++) {
      routes.push(
        createRandomRoute(
          `r_${i}_${r}`,
          `Route ${r + 1}`,
          ROUTE_COLORS[r % ROUTE_COLORS.length],
          graph
        )
      );
    }
    population.push({
      id: `gen0_rand_${i}`,
      routes,
      fitnessScore: 0
    });
  }

  return population;
}

/**
 * Applies genetic algorithm mutations to a candidate network.
 */
export function mutateGenome(
  genome: NetworkGenome,
  graph: KelantanGraphData,
  mutationRate = 0.35
): NetworkGenome {
  const routes: BusRoute[] = JSON.parse(JSON.stringify(genome.routes));

  // 1. Add Route Mutation
  if (Math.random() < mutationRate && routes.length < 9) {
    const newId = `r_mut_${Date.now()}_${Math.floor(Math.random() * 100)}`;
    routes.push(
      createRandomRoute(
        newId,
        `Route ${routes.length + 1}`,
        ROUTE_COLORS[routes.length % ROUTE_COLORS.length],
        graph
      )
    );
  }

  // 2. Remove Route Mutation
  if (Math.random() < mutationRate * 0.5 && routes.length > 2) {
    const removeIdx = Math.floor(Math.random() * routes.length);
    routes.splice(removeIdx, 1);
  }

  // 3. Extend/Shorten/Tweak Route Stops & Frequency Mutations
  routes.forEach(route => {
    if (Math.random() < mutationRate) {
      const choice = Math.random();
      
      // Extend route with connected node
      if (choice < 0.35 && route.nodeIds.length > 0) {
        const lastNodeId = route.nodeIds[route.nodeIds.length - 1];
        const neighbors = graph.edges.filter(e => e.from === lastNodeId).map(e => e.to);
        if (neighbors.length > 0) {
          const nextNode = neighbors[Math.floor(Math.random() * neighbors.length)];
          if (!route.nodeIds.includes(nextNode)) {
            route.nodeIds.push(nextNode);
          }
        }
      }
      // Shorten route
      else if (choice < 0.60 && route.nodeIds.length > 3) {
        route.nodeIds.pop();
      }
      // Tweak headway/bus frequency
      else {
        const busDelta = Math.random() > 0.5 ? 1 : -1;
        route.busCount = Math.max(2, Math.min(12, route.busCount + busDelta));
        route.headwayMinutes = Math.floor(60 / route.busCount);
      }
    }
  });

  return {
    id: `mut_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
    routes,
    fitnessScore: 0
  };
}

/**
 * Performs Crossover between two parent network genomes.
 */
export function crossoverGenomes(
  parentA: NetworkGenome,
  parentB: NetworkGenome
): NetworkGenome {
  const cutA = Math.floor(parentA.routes.length / 2);
  const cutB = Math.floor(parentB.routes.length / 2);

  const childRoutes = [
    ...JSON.parse(JSON.stringify(parentA.routes.slice(0, cutA))),
    ...JSON.parse(JSON.stringify(parentB.routes.slice(cutB)))
  ];

  return {
    id: `cross_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
    routes: childRoutes,
    fitnessScore: 0
  };
}

/**
 * Runs a single GA generation step.
 * Returns updated population, best genome, metrics, and DRT recommendations for killed routes.
 */
export function runEvolutionGeneration(
  population: NetworkGenome[],
  graph: KelantanGraphData,
  odTrips: ODTrip[],
  anchors: DemandAnchor[] = DEMAND_ANCHORS,
  mutationRate = 0.35,
  activeShocks?: ShockState
): {
  nextPopulation: NetworkGenome[];
  bestGenome: NetworkGenome;
  bestMetrics: EvaluationMetrics;
  drtRecommendations: { anchorId: string; anchorName: string; jajahan: string; lat: number; lng: number; message: string }[];
} {
  // Evaluate fitness for all individuals in population
  const evaluatedPop = population.map(genome => {
    const { compositeFitness, metrics } = evaluateGenomeRobust(genome, graph, odTrips, anchors, activeShocks);
    return {
      genome: { ...genome, fitnessScore: compositeFitness },
      metrics
    };
  });

  // Sort descending by fitness score
  evaluatedPop.sort((a, b) => b.genome.fitnessScore - a.genome.fitnessScore);

  const bestIndividual = evaluatedPop[0];
  const popSize = population.length;

  // Elitism: Preserve top 10% unchanged
  const eliteCount = Math.max(2, Math.floor(popSize * 0.10));
  const nextPopulation: NetworkGenome[] = evaluatedPop.slice(0, eliteCount).map(e => e.genome);

  // Tournament selection and reproduction to fill remaining population
  while (nextPopulation.length < popSize) {
    // Select parents via 3-way tournament selection
    const p1 = tournamentSelect(evaluatedPop);
    const p2 = tournamentSelect(evaluatedPop);

    let child = crossoverGenomes(p1.genome, p2.genome);
    child = mutateGenome(child, graph, mutationRate);

    nextPopulation.push(child);
  }

  // KILLED-ROUTE DRT DETECTOR:
  // Identify anchors/zones unserved by the best fixed bus routes
  const killedSet = new Set(bestIndividual.metrics.killedAnchorIds);
  const drtRecommendations = anchors
    .filter(a => killedSet.has(a.id))
    .map(a => ({
      anchorId: a.id,
      anchorName: a.name,
      jajahan: a.jajahan,
      lat: a.lat,
      lng: a.lng,
      message: `no fixed route — recommend demand-responsive van (${a.jajahan})`
    }));

  return {
    nextPopulation,
    bestGenome: bestIndividual.genome,
    bestMetrics: bestIndividual.metrics,
    drtRecommendations
  };
}

function tournamentSelect(evaluatedPop: { genome: NetworkGenome }[]): { genome: NetworkGenome } {
  const k = 3;
  let best = evaluatedPop[Math.floor(Math.random() * evaluatedPop.length)];
  for (let i = 1; i < k; i++) {
    const contestant = evaluatedPop[Math.floor(Math.random() * evaluatedPop.length)];
    if (contestant.genome.fitnessScore > best.genome.fitnessScore) {
      best = contestant;
    }
  }
  return best;
}
