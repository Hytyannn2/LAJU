'use client';

import { useEffect, useRef } from 'react';
import { DemandAnchor } from '@/data/laju/demandAnchors';
import { BusRoute, KelantanGraphData, ShockState } from '@/lib/laju/types';

interface LajuMapProps {
  graph: KelantanGraphData;
  anchors: DemandAnchor[];
  routes: BusRoute[];
  drtRecommendations: { anchorId: string; anchorName: string; jajahan: string; lat: number; lng: number; message: string }[];
  activeShocks: ShockState;
}

export default function LajuMap({
  graph,
  anchors,
  routes,
  drtRecommendations,
  activeShocks,
}: LajuMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const layerGroupRef = useRef<any>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Dynamically import Leaflet on client side
    import('leaflet').then(L => {
      if (!mapInstanceRef.current && mapContainerRef.current) {
        // Initialize map centered over Kelantan state
        const map = L.map(mapContainerRef.current, {
          center: [5.72, 102.15],
          zoom: 9,
          zoomControl: false,
          attributionControl: false,
        });

        // CartoDB Dark Matter tiles
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
          maxZoom: 18,
          subdomains: 'abcd',
        }).addTo(map);

        L.control.zoom({ position: 'topright' }).addTo(map);

        mapInstanceRef.current = map;
        layerGroupRef.current = L.layerGroup().addTo(map);
      }
    });

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update map features whenever routes, DRT recommendations, or shocks change
  useEffect(() => {
    if (!mapInstanceRef.current || !layerGroupRef.current) return;

    import('leaflet').then(L => {
      const layerGroup = layerGroupRef.current;
      layerGroup.clearLayers();

      const nodeMap = new Map(graph.nodes.map(n => [n.id, n]));

      // 1. Draw Skeleton Road Network (Subtle Dark Polylines)
      const graphEdgesSet = new Set<string>();
      graph.edges.forEach(edge => {
        const edgeKey = [edge.from, edge.to].sort().join('--');
        if (graphEdgesSet.has(edgeKey)) return;
        graphEdgesSet.add(edgeKey);

        const fromNode = nodeMap.get(edge.from);
        const toNode = nodeMap.get(edge.to);
        if (fromNode && toNode) {
          const isBlocked = activeShocks.monsoonClosure && 
            (edge.from.includes('sy_bridge') || edge.to.includes('sy_bridge') ||
             (edge.from.includes('kuala_krai') && edge.to.includes('chiku')));

          L.polyline(
            [[fromNode.lat, fromNode.lng], [toNode.lat, toNode.lng]],
            {
              color: isBlocked ? '#ef4444' : '#334155',
              weight: isBlocked ? 3 : 1,
              dashArray: isBlocked ? '6, 6' : undefined,
              opacity: isBlocked ? 0.9 : 0.45,
            }
          ).addTo(layerGroup);
        }
      });

      // 2. Draw Active Fixed-Route Bus Network Polylines
      routes.forEach(route => {
        const latLngs: [number, number][] = [];
        route.nodeIds.forEach(nodeId => {
          const node = nodeMap.get(nodeId);
          if (node) {
            latLngs.push([node.lat, node.lng]);
          }
        });

        if (latLngs.length > 1) {
          // Route polyline with glow effect
          const routePolyline = L.polyline(latLngs, {
            color: route.color,
            weight: 4,
            opacity: 0.85,
            lineCap: 'round',
            lineJoin: 'round',
          }).addTo(layerGroup);

          routePolyline.bindTooltip(
            `<div class="text-xs font-semibold px-2 py-1 bg-slate-900 text-white rounded border border-slate-700 shadow">
              <span style="color:${route.color}">●</span> ${route.name}<br/>
              <span class="text-slate-400">Headway: ${route.headwayMinutes} min | Buses: ${route.busCount}</span>
            </div>`,
            { sticky: true, opacity: 0.95 }
          );

          // Route stop dots
          latLngs.forEach(([lat, lng]) => {
            L.circleMarker([lat, lng], {
              radius: 4,
              fillColor: route.color,
              fillOpacity: 1,
              color: '#ffffff',
              weight: 1.5,
            }).addTo(layerGroup);
          });
        }
      });

      // 3. Draw Demand Anchors (Market, Hospital, School, Commercial)
      anchors.forEach(anchor => {
        let categoryColor = '#94a3b8';
        if (anchor.category === 'market') categoryColor = '#eab308'; // Yellow
        if (anchor.category === 'hospital') categoryColor = '#ef4444'; // Red
        if (anchor.category === 'school_uni') categoryColor = '#3b82f6'; // Blue
        if (anchor.category === 'commercial') categoryColor = '#ec4899'; // Pink
        if (anchor.category === 'transit_hub') categoryColor = '#10b981'; // Green

        // Special highlight for AEON Mall KB during AEON event shock
        const isAeonEvent = activeShocks.aeonEvent && anchor.id === 'kb_02';

        const marker = L.circleMarker([anchor.lat, anchor.lng], {
          radius: isAeonEvent ? 9 : 5,
          fillColor: isAeonEvent ? '#f43f5e' : categoryColor,
          fillOpacity: 0.8,
          color: isAeonEvent ? '#ffffff' : '#0f172a',
          weight: isAeonEvent ? 3 : 1,
        }).addTo(layerGroup);

        marker.bindPopup(
          `<div class="p-1 text-slate-800 text-xs">
            <strong class="font-bold text-slate-900">${anchor.name}</strong><br/>
            <span class="text-slate-600">Jajahan: ${anchor.jajahan}</span><br/>
            <span class="text-slate-600">Attractor Weight: ${anchor.weight} ${isAeonEvent ? '⚡ (8x Shock Boost)' : ''}</span>
          </div>`
        );
      });

      // 4. Draw KILLED ZONES / DRT RECOMMENDATIONS (Hollow Amber Rings)
      const drtIcon = L.divIcon({
        className: 'drt-custom-marker',
        html: `<div class="drt-pulse-ring"></div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      drtRecommendations.forEach(drt => {
        const drtMarker = L.marker([drt.lat, drt.lng], { icon: drtIcon }).addTo(layerGroup);

        drtMarker.bindTooltip(
          `<div class="text-xs font-semibold px-2.5 py-1.5 bg-amber-950 text-amber-200 rounded border border-amber-500/40 shadow-lg">
            <span class="text-amber-400">⚠️ DRT Equity Floor Triggered</span><br/>
            <span class="text-white">${drt.anchorName} (${drt.jajahan})</span><br/>
            <span class="text-amber-300 italic font-mono text-[11px]">${drt.message}</span>
          </div>`,
          { sticky: true, opacity: 0.95 }
        );
      });
    });
  }, [graph, anchors, routes, drtRecommendations, activeShocks]);

  return (
    <div className="relative w-full h-full dark-map">
      <div ref={mapContainerRef} className="w-full h-full z-0" />
    </div>
  );
}
