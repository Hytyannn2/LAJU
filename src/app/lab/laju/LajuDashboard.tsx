'use client';

import { useEffect, useRef, useState } from 'react';
import { DEMAND_ANCHORS } from '@/data/laju/demandAnchors';
import graphData from '@/data/laju/kelantanGraph.json';
import { BusRoute, DayOfWeek, EvaluationMetrics, KelantanGraphData, NetworkGenome, ShockState } from '@/lib/laju/types';
import { rlDispatcherStub } from '@/lib/laju/rlDispatcher';
import FitnessSparkline from './FitnessSparkline';
import LajuMap from './LajuMap';
import { Play, Pause, RotateCcw, Zap, Waves, Calendar, ShieldAlert, Bus, Layers, Activity, TrendingUp } from 'lucide-react';

const graph = graphData as unknown as KelantanGraphData;

export default function LajuDashboard() {
  const [isRunning, setIsRunning] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [speedMultiplier, setSpeedMultiplier] = useState(2); // 1x, 2x, 5x, Max
  const [mutationRate, setMutationRate] = useState(0.35);

  const [activeShocks, setActiveShocks] = useState<ShockState>({
    aeonEvent: false,
    monsoonClosure: false,
    hariRayaSurge: false,
  });

  const [selectedDay, setSelectedDay] = useState<DayOfWeek>('Friday');
  const [selectedHour, setSelectedHour] = useState(18); // 18:00 Friday Night Market surge

  const [bestGenome, setBestGenome] = useState<NetworkGenome | null>(null);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [fitnessHistory, setFitnessHistory] = useState<number[]>([]);
  const [drtRecommendations, setDrtRecommendations] = useState<{ anchorId: string; anchorName: string; jajahan: string; lat: number; lng: number; message: string }[]>([]);

  const workerRef = useRef<Worker | null>(null);
  const isRunningRef = useRef(isRunning);
  const speedRef = useRef(speedMultiplier);
  const shocksRef = useRef(activeShocks);
  const mutationRateRef = useRef(mutationRate);

  isRunningRef.current = isRunning;
  speedRef.current = speedMultiplier;
  shocksRef.current = activeShocks;
  mutationRateRef.current = mutationRate;

  // Initialize Web Worker for background GA execution
  useEffect(() => {
    // Dedicated Worker initialization
    const worker = new Worker(new URL('../../../workers/lajuWorker.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const { type, generation: gen, bestGenome: genome, bestMetrics: met, drtRecommendations: drts } = e.data;

      if (type === 'INIT_COMPLETE') {
        console.log('[LAJU Worker] Initialized population.');
      } else if (type === 'GENERATION_RESULT') {
        setGeneration(gen);

        // Apply stub RL dispatcher frequency tuning if trained hook present
        const tunedGenome = {
          ...genome,
          routes: rlDispatcherStub.tuneNetworkFrequencies(genome.routes, met)
        };

        setBestGenome(tunedGenome);
        setMetrics(met);
        setDrtRecommendations(drts || []);
        setFitnessHistory(prev => [...prev.slice(-40), genome.fitnessScore]);

        // Schedule next evolution step if running
        if (isRunningRef.current) {
          const delay = speedRef.current >= 5 ? 10 : Math.max(50, 400 / speedRef.current);
          setTimeout(() => {
            if (workerRef.current && isRunningRef.current) {
              workerRef.current.postMessage({
                type: 'STEP',
                data: {
                  activeShocks: shocksRef.current,
                  mutationRate: mutationRateRef.current
                }
              });
            }
          }, delay);
        }
      }
    };

    worker.postMessage({
      type: 'INIT',
      data: {
        graph,
        popSize: 60,
        mutationRate: 0.35,
        aeonBoost: activeShocks.aeonEvent
      }
    });

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Handle Play/Pause
  const toggleEvolution = () => {
    const nextState = !isRunning;
    setIsRunning(nextState);

    if (nextState && workerRef.current) {
      workerRef.current.postMessage({
        type: 'STEP',
        data: {
          activeShocks,
          mutationRate
        }
      });
    }
  };

  // Reset Evolution
  const resetEvolution = () => {
    setIsRunning(false);
    setGeneration(0);
    setFitnessHistory([]);
    setBestGenome(null);
    setMetrics(null);
    setDrtRecommendations([]);

    if (workerRef.current) {
      workerRef.current.postMessage({
        type: 'INIT',
        data: {
          graph,
          popSize: 60,
          mutationRate,
          aeonBoost: activeShocks.aeonEvent
        }
      });
    }
  };

  // Shock Toggles
  const toggleShock = (key: keyof ShockState) => {
    setActiveShocks(prev => {
      const updated = { ...prev, [key]: !prev[key] };

      // Trigger step immediate recalculation if worker active
      if (workerRef.current && isRunning) {
        workerRef.current.postMessage({
          type: 'STEP',
          data: {
            activeShocks: updated,
            mutationRate
          }
        });
      }

      return updated;
    });
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden flex flex-col bg-[#090d16] text-slate-100 font-sans">
      {/* Top Bar Header */}
      <header className="h-14 px-5 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold font-mono">
            🚏
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-wide">
                PROJECT LAJU <span className="text-xs font-mono font-normal text-emerald-400 px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800/50 ml-1">v1.0.0 Lab</span>
              </h1>
              <span className="text-[11px] font-medium text-slate-400 border border-slate-800 bg-slate-900 px-2 py-0.5 rounded-full">
                Kelantan State Transit AI
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Dasymetric OD • BPR Congestion • GA Evolution Engine • DRT Equity Floor
            </p>
          </div>
        </div>

        {/* Temporal Profile Indicator */}
        <div className="hidden lg:flex items-center gap-4 bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
          <div className="flex items-center gap-1.5 text-slate-300">
            <Calendar className="w-3.5 h-3.5 text-emerald-400" />
            <select
              value={selectedDay}
              onChange={e => setSelectedDay(e.target.value as DayOfWeek)}
              className="bg-slate-950 border border-slate-800 text-slate-200 rounded px-1.5 py-0.5 text-xs focus:outline-none"
            >
              <option value="Friday">Friday (Jumaat - Holy Day)</option>
              <option value="Saturday">Saturday (Sabtu - Weekend)</option>
              <option value="Sunday">Sunday (Weekday)</option>
              <option value="Monday">Monday (Weekday)</option>
            </select>
          </div>

          <div className="text-slate-400 border-l border-slate-800 pl-3">
            Simulated Hour: <span className="font-mono text-emerald-400 font-semibold">{selectedHour}:00</span>
            {selectedDay === 'Friday' && selectedHour >= 12 && selectedHour < 15 && (
              <span className="ml-2 text-[10px] bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded font-mono">
                🕌 Solat Jumaat Dip (0.15x)
              </span>
            )}
            {selectedDay === 'Friday' && selectedHour >= 17 && (
              <span className="ml-2 text-[10px] bg-purple-950 text-purple-300 border border-purple-800 px-1.5 py-0.5 rounded font-mono">
                🌙 Night Market Surge (1.65x)
              </span>
            )}
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={toggleEvolution}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all shadow-lg ${
              isRunning
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-emerald-600 text-white hover:bg-emerald-500 border border-emerald-400/50'
            }`}
          >
            {isRunning ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            {isRunning ? 'Pause Sim' : 'Evolve GA'}
          </button>

          <button
            onClick={resetEvolution}
            className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition"
            title="Reset GA Population"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Container */}
      <div className="relative flex-1 w-full h-full overflow-hidden">
        {/* Full-bleed Leaflet Map */}
        <LajuMap
          graph={graph}
          anchors={DEMAND_ANCHORS}
          routes={bestGenome?.routes || []}
          drtRecommendations={drtRecommendations}
          activeShocks={activeShocks}
        />

        {/* LEFT HUD PANEL: Evolution Dashboard */}
        <div className="absolute top-4 left-4 z-10 w-80 glass-panel rounded-xl p-4 flex flex-col gap-4 text-xs">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <div>
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">GA Optimization</span>
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-black font-mono text-emerald-400">Gen #{generation}</span>
                <span className="text-slate-400 text-[11px]">Pop: 60 candidate networks</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Best Fitness</span>
              <div className="text-lg font-bold font-mono text-white">
                {bestGenome ? bestGenome.fitnessScore.toFixed(4) : '0.0000'}
              </div>
            </div>
          </div>

          {/* Fitness Sparkline */}
          <div>
            <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
              <span className="flex items-center gap-1"><TrendingUp className="w-3 h-3 text-emerald-400" /> Fitness Curve</span>
              <span className="font-mono text-emerald-400">
                {fitnessHistory.length > 0 ? `+${((fitnessHistory[fitnessHistory.length - 1] - fitnessHistory[0]) / Math.max(0.001, fitnessHistory[0]) * 100).toFixed(1)}%` : '0.0%'}
              </span>
            </div>
            <FitnessSparkline history={fitnessHistory} width={288} height={45} />
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-[cols-2] gap-2 pt-1 border-t border-slate-800/80">
            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-lg">
              <div className="text-[10px] text-slate-400">Coverage</div>
              <div className="text-sm font-bold font-mono text-white">
                {metrics ? `${metrics.coveragePercent}%` : '--'}
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-lg">
              <div className="text-[10px] text-slate-400">Avg Wait Time</div>
              <div className="text-sm font-bold font-mono text-amber-400">
                {metrics ? `${metrics.avgWaitMinutes} m` : '--'}
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-lg">
              <div className="text-[10px] text-slate-400">Avg Travel Time</div>
              <div className="text-sm font-bold font-mono text-cyan-400">
                {metrics ? `${metrics.avgTravelMinutes} m` : '--'}
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-2 rounded-lg">
              <div className="text-[10px] text-slate-400">Daily Fleet Cost</div>
              <div className="text-sm font-bold font-mono text-emerald-400">
                {metrics ? `RM ${metrics.fleetCostRM.toLocaleString()}` : '--'}
              </div>
            </div>
          </div>

          {/* Equity Index Score */}
          <div className="bg-slate-900/90 border border-slate-800 p-2.5 rounded-lg flex items-center justify-between">
            <div>
              <div className="text-[10px] text-slate-400">Jajahan Equity Index</div>
              <div className="text-xs font-semibold text-slate-200">Across all 11 Kelantan Districts</div>
            </div>
            <div className="text-base font-black font-mono text-purple-400">
              {metrics ? `${metrics.equityIndex}%` : '--'}
            </div>
          </div>

          {/* Evolution Settings */}
          <div className="flex flex-col gap-2 pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Evolution Speed</span>
              <div className="flex gap-1">
                {[1, 2, 5].map(s => (
                  <button
                    key={s}
                    onClick={() => setSpeedMultiplier(s)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono transition ${
                      speedMultiplier === s
                        ? 'bg-emerald-600 text-white font-bold'
                        : 'bg-slate-900 text-slate-400 border border-slate-800'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Mutation Rate: <strong className="font-mono text-slate-200">{(mutationRate * 100).toFixed(0)}%</strong></span>
              <input
                type="range"
                min="0.10"
                max="0.75"
                step="0.05"
                value={mutationRate}
                onChange={e => setMutationRate(parseFloat(e.target.value))}
                className="w-24 accent-emerald-500"
              />
            </div>
          </div>
        </div>

        {/* RIGHT HUD PANEL: Shock Modes & Killed-Route DRT Van Drawer */}
        <div className="absolute top-4 right-4 z-10 w-80 glass-panel rounded-xl p-4 flex flex-col gap-4 text-xs">
          {/* Shock Modes Controls */}
          <div>
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200 uppercase tracking-wider mb-2">
              <Zap className="w-3.5 h-3.5 text-amber-400" /> Shock Mode Simulators
            </div>
            <p className="text-[11px] text-slate-400 mb-3">
              Final fitness = 0.7*normal + 0.3*worstShock. Networks train for chaos.
            </p>

            <div className="flex flex-col gap-2">
              <button
                onClick={() => toggleShock('aeonEvent')}
                className={`flex items-center justify-between px-3 py-2 rounded-lg border transition ${
                  activeShocks.aeonEvent
                    ? 'bg-rose-950/80 border-rose-500 text-rose-200'
                    : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-xs">
                  ⚡ AEON KB Event (8x Demand)
                </span>
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${activeShocks.aeonEvent ? 'bg-rose-600 text-white' : 'bg-slate-800'}`}>
                  {activeShocks.aeonEvent ? 'ON' : 'OFF'}
                </span>
              </button>

              <button
                onClick={() => toggleShock('monsoonClosure')}
                className={`flex items-center justify-between px-3 py-2 rounded-lg border transition ${
                  activeShocks.monsoonClosure
                    ? 'bg-blue-950/80 border-blue-500 text-blue-200'
                    : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-xs">
                  <Waves className="w-3.5 h-3.5 text-blue-400" /> Monsoon Bridge Closure
                </span>
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${activeShocks.monsoonClosure ? 'bg-blue-600 text-white' : 'bg-slate-800'}`}>
                  {activeShocks.monsoonClosure ? 'ON' : 'OFF'}
                </span>
              </button>

              <button
                onClick={() => toggleShock('hariRayaSurge')}
                className={`flex items-center justify-between px-3 py-2 rounded-lg border transition ${
                  activeShocks.hariRayaSurge
                    ? 'bg-emerald-950/80 border-emerald-500 text-emerald-200'
                    : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-xs">
                  🌙 Hari Raya Festivity Surge (3x)
                </span>
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${activeShocks.hariRayaSurge ? 'bg-emerald-600 text-white' : 'bg-slate-800'}`}>
                  {activeShocks.hariRayaSurge ? 'ON' : 'OFF'}
                </span>
              </button>
            </div>
          </div>

          {/* KILLED ZONES / DRT RECOMMENDATIONS DRAWER */}
          <div className="pt-2 border-t border-slate-800/80 flex-1 overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <span className="flex items-center gap-1.5 text-xs font-bold text-amber-400 uppercase tracking-wider">
                <ShieldAlert className="w-3.5 h-3.5" /> DRT Van Recommendations ({drtRecommendations.length})
              </span>
            </div>

            <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
              {drtRecommendations.length === 0 ? (
                <div className="p-3 text-center text-slate-500 italic bg-slate-900/50 rounded border border-slate-800">
                  All demand anchors currently served by fixed routes.
                </div>
              ) : (
                drtRecommendations.map(drt => (
                  <div key={drt.anchorId} className="bg-amber-950/30 border border-amber-500/30 p-2 rounded text-[11px]">
                    <div className="font-semibold text-amber-200">{drt.anchorName}</div>
                    <div className="flex items-center justify-between text-slate-400 text-[10px] mt-0.5">
                      <span>Jajahan: {drt.jajahan}</span>
                      <span className="text-amber-400 font-mono italic">Demand-Responsive Van</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* RL Dispatcher Interface Stub status */}
          <div className="pt-2 border-t border-slate-800/80 text-[10px] text-slate-500 flex items-center justify-between font-mono">
            <span>RL Dispatcher: {rlDispatcherStub.name}</span>
            <span className="text-emerald-400 bg-emerald-950 px-1.5 py-0.5 rounded border border-emerald-800">STUB OK</span>
          </div>
        </div>
      </div>
    </div>
  );
}
