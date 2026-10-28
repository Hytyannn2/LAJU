"use client";

import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useMemo, useState } from "react";
import { CircleMarker, LayerGroup, LayersControl, MapContainer, Pane, Polygon, Polyline, Popup, TileLayer, Tooltip, useMapEvents } from "react-leaflet";
import type { BBox, Design, LatLng, PlannerStop } from "./Designer";

// Kelantan only: tiles outside this box are never requested and the map cannot leave it.
const KELANTAN: [LatLng, LatLng] = [
  [4.45, 101.2],
  [6.35, 102.8],
];
const WORLD: LatLng[] = [
  [-85, -180],
  [-85, 180],
  [85, 180],
  [85, -180],
];

const congestionColor = (c: number) => (c < 0.25 ? "#22c55e" : c < 0.5 ? "#eab308" : c < 0.75 ? "#f97316" : "#ef4444");

function Events({ onView, focus }: { onView: (b: BBox) => void; focus: BBox | null }) {
  const map = useMapEvents({
    moveend: () => {
      const b = map.getBounds();
      onView([b.getSouth(), b.getWest(), b.getNorth(), b.getEast()]);
    },
  });
  useEffect(() => {
    // Leaflet measures its box once; re-measure whenever the layout (stylesheet, sidebar) resizes it.
    const ro = new ResizeObserver(() => {
      map.invalidateSize();
      const b = map.getBounds();
      onView([b.getSouth(), b.getWest(), b.getNorth(), b.getEast()]);
    });
    ro.observe(map.getContainer());
    return () => ro.disconnect();
  }, [map, onView]);
  useEffect(() => {
    if (focus) map.fitBounds([[focus[0], focus[1]], [focus[2], focus[3]]]);
  }, [map, focus]);
  return null;
}

function OutsideKelantan() {
  const [rings, setRings] = useState<LatLng[][]>([]);
  useEffect(() => {
    fetch("/engine/kelantan")
      .then((r) => (r.ok ? r.json() : null))
      .then((g) => {
        if (!g) return;
        const polys: number[][][][] = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
        setRings(polys.map((p) => p[0].map(([lng, lat]) => [lat, lng] as LatLng)));
      })
      .catch(() => {});
  }, []);
  if (!rings.length) return null;
  return (
    <Polygon
      positions={[WORLD, ...rings]}
      interactive={false}
      pathOptions={{ color: "#64748b", weight: 1.5, fillColor: "#000", fillOpacity: 0.55 }}
    />
  );
}

const busCanvas = L.canvas({ pane: "buses" }); // hundreds of moving dots: one canvas, above the routes and stops

// Point at `d` metres along a path, given its cumulative metres `cum`.
function along(path: LatLng[], cum: number[], d: number): LatLng {
  let k = 1;
  while (k < cum.length - 1 && cum[k] < d) k++;
  const f = cum[k] > cum[k - 1] ? (d - cum[k - 1]) / (cum[k] - cum[k - 1]) : 0;
  return [path[k - 1][0] + f * (path[k][0] - path[k - 1][0]), path[k - 1][1] + f * (path[k][1] - path[k - 1][1])];
}

// Buses driving each route out and back at its planned minutes, spaced by its headway. 1 s on screen = 1 min.
// ponytail: no layover and no traffic; the SUMO run (Simulate peak hour) is the realistic one.
function Buses({ design, visible }: { design: Design; visible: (i: number) => boolean }) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const t0 = performance.now();
    const id = window.setInterval(() => setT((performance.now() - t0) / 1000), 100);
    return () => clearInterval(id);
  }, []);
  const cums = useMemo(
    () =>
      design.routes.map((r) =>
        r.path.reduce<number[]>((c, p, k) => {
          const q = r.path[k - 1];
          c.push(k ? c[k - 1] + Math.hypot((p[0] - q[0]) * 110570, (p[1] - q[1]) * 110800) : 0);
          return c;
        }, []),
      ),
    [design],
  );
  const m = design.metrics;
  return design.routes.flatMap((r, i) => {
    if (!visible(i) || r.path.length < 2) return [];
    const out = Math.max(m.route_min[i], 1);
    const back = Math.max(m.route_min_back[i], 1);
    const n = Math.max(1, Math.round((out + back) / m.headway[i]));
    const cum = cums[i];
    const len = cum[cum.length - 1];
    return Array.from({ length: n }, (_, k) => {
      const p = (t + (k * (out + back)) / n) % (out + back); // minutes into this bus's round trip
      const d = p < out ? (p / out) * len : (1 - (p - out) / back) * len;
      return (
        <CircleMarker
          key={`b${i}-${k}`}
          center={along(r.path, cum, d)}
          radius={5}
          interactive={false}
          renderer={busCanvas}
          pathOptions={{ color: "#000", weight: 2, fillColor: "#facc15", fillOpacity: 1 }} // bus yellow: stands out on any route colour
        />
      );
    });
  });
}

export default function MapView({
  design,
  colors,
  planner,
  locked,
  showTraffic,
  showBuses,
  visible,
  focus,
  onView,
}: {
  design: Design | null;
  colors: string[];
  planner: PlannerStop[];
  locked: Set<number>;
  showTraffic: boolean;
  showBuses: boolean;
  visible: (i: number) => boolean; // route filter: only these routes, their stops and buses are drawn
  focus: BBox | null;
  onView: (b: BBox) => void;
}) {
  const names = useMemo(() => new Map(design?.stops.map((s) => [s.id, s.name])), [design]);
  return (
    <MapContainer
      center={[6.1, 102.25]}
      zoom={11}
      minZoom={8}
      maxBounds={KELANTAN}
      maxBoundsViscosity={1}
      className="h-full w-full"
    >
      <LayersControl position="topright">
        {/* Esri World Imagery + place names: free with attribution while developing; selling LAJU needs a licence (PLAN section 6). */}
        <LayersControl.BaseLayer checked name="Satellite">
          <LayerGroup>
            <TileLayer
              attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              bounds={KELANTAN}
              noWrap
            />
            <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}" bounds={KELANTAN} noWrap />
          </LayerGroup>
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Dark streets">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            className="dark-tiles"
            bounds={KELANTAN}
            noWrap
          />
        </LayersControl.BaseLayer>
      </LayersControl>
      <OutsideKelantan />
      <Events onView={onView} focus={focus} />
      {showTraffic &&
        design?.sumo?.congestion.map((e, i) => (
          <Polyline key={`c${i}`} positions={e.path} interactive={false} pathOptions={{ color: congestionColor(e.c), weight: 3, opacity: 0.8 }} />
        ))}
      {design?.routes.map(
        (r, i) =>
          visible(i) && (
            <Polyline
              key={`r${i}`}
              positions={r.path}
              pathOptions={{ color: colors[i % colors.length], weight: 5, opacity: 0.85, dashArray: locked.has(i) ? "10 6" : undefined }}
            >
              <Tooltip sticky>Route {i + 1}</Tooltip>
            </Polyline>
          ),
      )}
      {design?.stops.map((s) => {
        const first = s.routes.find(visible);
        return (
          first !== undefined && (
            <CircleMarker
              key={s.id}
              center={[s.lat, s.lng]}
              radius={s.key ? 6 : 3.5}
              bubblingMouseEvents={false}
              pathOptions={{ color: colors[first % colors.length], weight: s.key ? 3 : 2, fillColor: s.key ? "#fff" : "#111", fillOpacity: 1 }}
            >
              <Tooltip direction="bottom">
                <b>{s.name}</b> · {s.key ? "key stop" : "stop"} on a {s.road}
                <br />
                Route {s.routes.map((r) => r + 1).join(", ")}
              </Tooltip>
              {/* No auto-pan: react-leaflet re-runs it on every re-render (training polls every 2 s), dragging the map back here. */}
              <Popup autoPan={false} maxWidth={360} maxHeight={320}>
                <b>{s.name}</b> · {s.key ? "key stop" : "stop"} on a {s.road}
                {s.routes.map((i) => {
                  const r = design.routes[i];
                  const at = r.stops.flatMap((id, k) => (id === s.id ? [k + 1] : [])); // twice on a route that comes back past it
                  return (
                    <div key={i} className="mt-1">
                      <span className="mr-1 inline-block h-2.5 w-2.5 rounded-full" style={{ background: colors[i % colors.length] }} />
                      <b>Route {i + 1}</b> ({names.get(r.stops[0])} → {names.get(r.stops[r.stops.length - 1])}): stop {at.join(" and ")} of{" "}
                      {r.stops.length} · a bus every {design.metrics.headway[i]} min, {design.metrics.buses[i]} buses on the route
                    </div>
                  );
                })}
                {s.routes.length > 1 && (
                  <div className="mt-1">
                    All its routes: about {Math.round(s.routes.reduce((n, i) => n + 60 / design.metrics.headway[i], 0))} buses an hour each way
                  </div>
                )}
              </Popup>
            </CircleMarker>
          )
        );
      })}
      {/* Click-through: the bus canvas covers the whole map, so it would take every click and hover meant for the stops. */}
      <Pane name="buses" style={{ zIndex: 450, pointerEvents: "none" }}>
        {showBuses && design && <Buses design={design} visible={visible} />}
      </Pane>
      {planner.map((s) => (
        <CircleMarker
          key={s.id}
          center={[s.lat, s.lng]}
          radius={8}
          bubblingMouseEvents={false}
          pathOptions={{ color: "#facc15", weight: 3, fillColor: "#facc15", fillOpacity: 0.35 }}
        >
          <Tooltip>{s.name || "Your stop"} · forced stop (moved to the nearest place a bus may stop)</Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
