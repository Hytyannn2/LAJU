import { BusRoute, EvaluationMetrics } from './types';

export interface RLState {
  routeId: string;
  currentHeadwayMinutes: number;
  passengerQueueLength: number;
  roadCongestionRatio: number;
  timeOfDayHour: number;
}

export interface RLAction {
  routeId: string;
  actionType: 'INCREASE_FREQUENCY' | 'DECREASE_FREQUENCY' | 'MAINTAIN' | 'EXPRESS_SKIP_STOPS';
  headwayDeltaMinutes: number;
}

export interface RLReward {
  routeId: string;
  rewardValue: number; // Reward derived from wait time reduction vs operating cost
}

export interface IRLDispatcher {
  name: string;
  version: string;
  isTrained: boolean;
  selectAction(state: RLState): RLAction;
  observeReward(reward: RLReward): void;
  tuneNetworkFrequencies(routes: BusRoute[], metrics: EvaluationMetrics): BusRoute[];
}

/**
 * Stubbed Reinforcement Learning Dispatcher for dynamic bus headway and real-time dispatching.
 * Ready for future Q-learning / Deep Q-Network (DQN) policy training inside the simulation loop.
 */
export class StubRLDispatcher implements IRLDispatcher {
  name = 'Kelantan-RL-Headway-Dispatcher-v0.1-Stub';
  version = '0.1.0-alpha';
  isTrained = false;

  selectAction(state: RLState): RLAction {
    // Stub heuristic decision rule
    if (state.passengerQueueLength > 50 && state.currentHeadwayMinutes > 10) {
      return {
        routeId: state.routeId,
        actionType: 'INCREASE_FREQUENCY',
        headwayDeltaMinutes: -5
      };
    } else if (state.passengerQueueLength < 10 && state.currentHeadwayMinutes < 40) {
      return {
        routeId: state.routeId,
        actionType: 'DECREASE_FREQUENCY',
        headwayDeltaMinutes: 5
      };
    }
    return {
      routeId: state.routeId,
      actionType: 'MAINTAIN',
      headwayDeltaMinutes: 0
    };
  }

  observeReward(reward: RLReward): void {
    // Stub logging for training loop
    if (process.env.NODE_ENV === 'development') {
      console.log(`[RL Dispatcher Stub] Observed reward ${reward.rewardValue} for route ${reward.routeId}`);
    }
  }

  tuneNetworkFrequencies(routes: BusRoute[], metrics: EvaluationMetrics): BusRoute[] {
    // Stub route headway adjustment hook
    return routes.map(route => {
      if (metrics.avgWaitMinutes > 25 && route.headwayMinutes > 15) {
        return {
          ...route,
          headwayMinutes: Math.max(10, route.headwayMinutes - 2),
          busCount: route.busCount + 1
        };
      }
      return route;
    });
  }
}

export const rlDispatcherStub = new StubRLDispatcher();
