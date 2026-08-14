import { DayOfWeek, ShockState } from './types';

/**
 * Returns hourly demand multiplier (0.0 to 2.0+) based on day of week and hour (0-23).
 * Encodes Kelantan's specific Jumaat-Sabtu weekend and Friday bimodal prayer schedule.
 */
export function getHourlyDemandMultiplier(day: DayOfWeek, hour: number): number {
  // --- FRIDAY (Jumaat: Kelantan Weekend Day 1 & Holy Day) ---
  if (day === 'Friday') {
    if (hour >= 7 && hour <= 10) return 1.45; // Morning market surge (Pasar Siti Khadijah)
    if (hour >= 11 && hour < 12) return 0.80; // Transition to prayer preparation
    if (hour >= 12 && hour < 15) return 0.15; // 12:00-14:45 Solat Jumaat prayer dip
    if (hour >= 15 && hour <= 16) return 0.90; // Post-prayer resumption
    if (hour >= 17 && hour <= 22) return 1.65; // Night market surge (Wakaf Che Yeh)
    return 0.30; // Late night
  }

  // --- SATURDAY (Sabtu: Kelantan Weekend Day 2) ---
  if (day === 'Saturday') {
    if (hour >= 10 && hour <= 21) return 1.50; // Continuous weekend shopping & leisure wave
    if (hour >= 7 && hour <= 9) return 0.85;
    return 0.35;
  }

  // --- SUNDAY - THURSDAY (Kelantan Weekdays / Working Days) ---
  if (hour >= 7 && hour <= 9) return 1.35; // Morning commute & school drop-off peak
  if (hour >= 12 && hour <= 14) return 0.95; // Lunch hour transit
  if (hour >= 17 && hour <= 19) return 1.40; // Evening work return peak
  if (hour >= 20 && hour <= 22) return 0.60;
  return 0.25; // Overnight low demand
}

/**
 * Calculates effective aggregate daily demand scale considering shock modes.
 */
export function getEffectiveDemandMultiplier(
  day: DayOfWeek = 'Friday',
  hour = 18,
  shocks?: Partial<ShockState>
): number {
  let baseMult = getHourlyDemandMultiplier(day, hour);

  if (shocks?.hariRayaSurge) {
    baseMult *= 3.0; // 3x state-wide festive travel surge
  }

  return parseFloat(baseMult.toFixed(2));
}
