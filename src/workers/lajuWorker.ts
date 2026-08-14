import { DEMAND_ANCHORS } from '../data/laju/demandAnchors';
import { initializePopulation, runEvolutionGeneration } from '../lib/laju/evolution';
import { generateODMatrix } from '../lib/laju/gravity';
import { DayOfWeek, EvaluationMetrics, KelantanGraphData, NetworkGenome, ODTrip, ShockState } from '../lib/laju/types';

let currentPopulation: NetworkGenome[] = [];
let cachedGraph: KelantanGraphData | null = null;
let cachedODTrips: ODTrip[] = [];
let currentGeneration = 0;

self.onmessage = function (event: MessageEvent) {
  const { type, data } = event.data;

  if (type === 'INIT') {
    cachedGraph = data.graph;
    cachedODTrips = generateODMatrix(cachedGraph!.nodes, DEMAND_ANCHORS, 50000, data.aeonBoost);
    currentPopulation = initializePopulation(data.popSize || 60, cachedGraph!);
    currentGeneration = 0;
    
    self.postMessage({
      type: 'INIT_COMPLETE',
      generation: 0,
      popSize: currentPopulation.length
    });
  }

  else if (type === 'STEP') {
    if (!cachedGraph || currentPopulation.length === 0) return;

    const { activeShocks, mutationRate } = data as {
      activeShocks: ShockState;
      mutationRate: number;
    };

    // Recompute OD matrix if AEON shock toggled
    if (activeShocks?.aeonEvent) {
      cachedODTrips = generateODMatrix(cachedGraph.nodes, DEMAND_ANCHORS, 50000, true);
    }

    currentGeneration += 1;

    const result = runEvolutionGeneration(
      currentPopulation,
      cachedGraph,
      cachedODTrips,
      DEMAND_ANCHORS,
      mutationRate || 0.35,
      activeShocks
    );

    currentPopulation = result.nextPopulation;

    self.postMessage({
      type: 'GENERATION_RESULT',
      generation: currentGeneration,
      bestGenome: result.bestGenome,
      bestMetrics: result.bestMetrics,
      drtRecommendations: result.drtRecommendations
    });
  }

  else if (type === 'RESET') {
    currentPopulation = [];
    currentGeneration = 0;
    self.postMessage({ type: 'RESET_COMPLETE' });
  }
};
