'use client';

interface FitnessSparklineProps {
  history: number[];
  width?: number;
  height?: number;
}

export default function FitnessSparkline({
  history,
  width = 240,
  height = 50,
}: FitnessSparklineProps) {
  if (!history || history.length < 2) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center text-xs text-slate-500 font-mono border border-dashed border-slate-800 rounded"
      >
        Initializing GA fitness curve...
      </div>
    );
  }

  const min = Math.min(...history);
  const max = Math.max(...history);
  const range = max - min || 0.01;

  const points = history.map((val, idx) => {
    const x = (idx / (history.length - 1)) * width;
    const y = height - ((val - min) / range) * (height - 8) - 4;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const areaPoints = `0,${height} ${points} ${width},${height}`;

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <linearGradient id="fitnessGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Gradient fill */}
        <polygon points={areaPoints} fill="url(#fitnessGradient)" />

        {/* Sparkline path */}
        <polyline
          fill="none"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />

        {/* Current peak point dot */}
        {history.length > 0 && (
          <circle
            cx={width}
            cy={height - ((history[history.length - 1] - min) / range) * (height - 8) - 4}
            r="3"
            fill="#10b981"
            className="animate-pulse"
          />
        )}
      </svg>
    </div>
  );
}
