"""Phase 5 — export de la visualisation du champ de bassins.

Calcule le champ de bassins de trois règles ECA (un régime chacune) et génère
une page HTML autonome: graphe de force D3, fond noir, couleurs émissives par
bassin, force d'ancrage qui sépare visuellement les bassins, sélecteur de régime."""

import json
from pathlib import Path

from ddlab_bench import computeBasinField
from ddlab_bench.viz import buildBasinGraph

N = 9
RULES = [(250, "ordonné"), (110, "complexe"), (30, "chaotique")]

HTML_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Champ de bassins d'attraction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
  :root {
    --bg: #07080f; --panel: rgba(14,16,32,0.92); --fg: #e2e8f0;
    --muted: #8893a8; --border: rgba(255,255,255,0.09);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: 'JetBrains Mono', monospace; overflow: hidden;
  }
  body::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 6;
    background: radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.62) 100%);
  }
  header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    padding: 18px 26px; display: flex; align-items: baseline; gap: 22px; flex-wrap: wrap;
    background: linear-gradient(180deg, rgba(7,8,15,0.96), rgba(7,8,15,0));
  }
  h1 {
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 16px;
    letter-spacing: 0.1em; text-transform: uppercase; color: #fff;
    text-shadow: 0 0 14px rgba(120,170,255,0.55);
  }
  .subtitle { color: var(--muted); font-size: 11.5px; }
  .controls { position: fixed; top: 60px; left: 26px; z-index: 10; display: flex; gap: 8px; }
  .rule-btn {
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
    background: var(--panel); color: var(--muted);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 13px; cursor: pointer;
    transition: color 180ms, border-color 180ms, box-shadow 180ms;
  }
  .rule-btn:hover { color: var(--fg); border-color: rgba(255,255,255,0.28); }
  .rule-btn.active { color: #fff; border-color: rgba(120,170,255,0.7);
    box-shadow: 0 0 16px rgba(120,170,255,0.32); }
  .panel {
    position: fixed; z-index: 10; background: var(--panel);
    border: 1px solid var(--border); border-radius: 9px; padding: 14px 16px;
    backdrop-filter: blur(6px); font-size: 11.5px;
  }
  #info { top: 60px; right: 26px; min-width: 200px; }
  #info .row { display: flex; justify-content: space-between; gap: 18px; padding: 3px 0; }
  #info .label { color: var(--muted); }
  #legend { bottom: 26px; left: 26px; }
  #legend .item { display: flex; align-items: center; gap: 9px; padding: 3.5px 0; }
  #legend .dot { width: 12px; height: 12px; border-radius: 50%; flex: none; }
  #legend .note { color: var(--muted); margin-top: 7px; padding-top: 7px;
    border-top: 1px solid var(--border); }
  .caption {
    position: fixed; bottom: 26px; right: 26px; z-index: 10;
    color: var(--muted); font-size: 11px; max-width: 332px; text-align: right;
    line-height: 1.65;
  }
  #tooltip {
    position: fixed; z-index: 20; pointer-events: none; opacity: 0;
    background: rgba(14,16,32,0.97); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 11px; font-size: 11px;
    line-height: 1.6; transition: opacity 120ms; white-space: nowrap;
  }
  svg { display: block; width: 100vw; height: 100vh; }
  .node { cursor: pointer; }
  @media (prefers-reduced-motion: reduce) { .rule-btn { transition: none; } }
</style>
</head>
<body>
<header>
  <h1>Champ de bassins d'attraction</h1>
  <span class="subtitle">DDLab sense-extraction benchmark · automate cellulaire élémentaire · n=9, 512 états</span>
</header>
<div class="controls" id="controls"></div>
<div class="panel" id="info"></div>
<div class="panel" id="legend">
  <div class="item"><span class="dot" style="background:#fff;box-shadow:0 0 9px #fff"></span> attracteur (cycle)</div>
  <div class="item"><span class="dot" style="background:#8893a8"></span> transitoire</div>
  <div class="item"><span class="dot" style="background:#4b5468"></span> garden-of-Eden</div>
  <div class="note">la couleur encode le bassin</div>
</div>
<div class="caption">Chaque nœud = un état ; chaque arête = une transition. Les cycles
attracteurs sont les moyeux lumineux, les états garden-of-Eden (sans passé) les
feuilles. Glisser un nœud, molette pour zoomer.</div>
<div id="tooltip"></div>
<svg id="canvas"></svg>
<script>
const DATASETS = __DATA__;
const PALETTE = ["#22d3ee","#e879f9","#a3e635","#fb923c","#a78bfa","#fb7185",
  "#fde047","#38bdf8","#34d399","#f87171","#60a5fa","#f472b6","#2dd4bf",
  "#fbbf24","#818cf8","#4ade80"];

const svg = d3.select("#canvas");
const tooltip = d3.select("#tooltip");
let width = window.innerWidth, height = window.innerHeight;
svg.attr("viewBox", [0, 0, width, height]);

const defs = svg.append("defs");
defs.append("filter").attr("id", "bloom")
  .attr("x", "-60%").attr("y", "-60%").attr("width", "220%").attr("height", "220%")
  .append("feGaussianBlur").attr("stdDeviation", 4);

const root = svg.append("g");
const edgeLayer = root.append("g");
const glowLayer = root.append("g").attr("filter", "url(#bloom)").attr("opacity", 0.5);
const nodeLayer = root.append("g");

const zoom = d3.zoom().scaleExtent([0.15, 6])
  .on("zoom", (event) => root.attr("transform", event.transform));
svg.call(zoom);

let simulation = null;

function kindRadius(kind) { return kind === "attractor" ? 7 : kind === "goe" ? 2.6 : 4; }
function kindOpacity(kind) { return kind === "attractor" ? 1 : kind === "goe" ? 0.42 : 0.82; }
function basinColor(d) { return PALETTE[d.basin % PALETTE.length]; }

function render(index) {
  const data = DATASETS[index];
  const nodes = data.nodes.map((d) => Object.assign({}, d));
  const edges = data.edges.map((d) => Object.assign({}, d));

  // ancre par bassin: chaque bassin est tiré vers un point d'un anneau,
  // ce qui sépare visuellement les bassins en grappes distinctes
  const basinIds = Array.from(new Set(nodes.map((d) => d.basin)));
  const anchors = {};
  const ringRadius = Math.min(width, height) * 0.33;
  basinIds.forEach((basin, i) => {
    const angle = (i / basinIds.length) * 2 * Math.PI;
    anchors[basin] = {
      x: width / 2 + ringRadius * Math.cos(angle),
      y: height / 2 + ringRadius * Math.sin(angle),
    };
  });

  edgeLayer.selectAll("*").remove();
  glowLayer.selectAll("*").remove();
  nodeLayer.selectAll("*").remove();

  const edgeSel = edgeLayer.selectAll("line").data(edges).join("line")
    .attr("stroke-width", 1);
  const glowSel = glowLayer.selectAll("circle").data(nodes).join("circle")
    .attr("r", (d) => kindRadius(d.kind) * 1.7)
    .attr("fill", basinColor);
  const nodeSel = nodeLayer.selectAll("circle").data(nodes).join("circle")
    .attr("class", "node")
    .attr("r", (d) => kindRadius(d.kind))
    .attr("fill", basinColor)
    .attr("fill-opacity", (d) => kindOpacity(d.kind))
    .attr("stroke", (d) => d.kind === "attractor" ? "#fff" : "none")
    .attr("stroke-width", (d) => d.kind === "attractor" ? 1 : 0)
    .on("mouseover", (event, d) => {
      tooltip.style("opacity", 1).html(
        "état " + d.id + "<br>bassin " + d.basin + " · " + d.kind
        + "<br>transitoire " + d.transient);
    })
    .on("mousemove", (event) => {
      tooltip.style("left", (event.clientX + 15) + "px")
        .style("top", (event.clientY + 15) + "px");
    })
    .on("mouseout", () => tooltip.style("opacity", 0));

  nodeSel.call(d3.drag()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }));

  function fitView() {
    const xs = nodes.map((d) => d.x), ys = nodes.map((d) => d.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min(width / (maxX - minX + 150),
      height / (maxY - minY + 150), 2.4);
    const tx = width / 2 - scale * (minX + maxX) / 2;
    const ty = height / 2 - scale * (minY + maxY) / 2;
    svg.transition().duration(700)
      .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id((d) => d.id).distance(11).strength(0.45))
    .force("charge", d3.forceManyBody().strength(-16))
    .force("x", d3.forceX((d) => anchors[d.basin].x).strength(0.22))
    .force("y", d3.forceY((d) => anchors[d.basin].y).strength(0.22))
    .force("collide", d3.forceCollide((d) => kindRadius(d.kind) + 1.4))
    .on("tick", () => {
      edgeSel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      glowSel.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      nodeSel.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    })
    .on("end", fitView);

  // arêtes teintées par bassin (source/target résolus après le forceLink)
  edgeSel.attr("stroke", (d) => basinColor(d.source)).attr("stroke-opacity", 0.34);

  d3.select("#info").html(
    "<div class='row'><span class='label'>règle</span><span>" + data.rule + "</span></div>"
    + "<div class='row'><span class='label'>régime</span><span>" + data.regime + "</span></div>"
    + "<div class='row'><span class='label'>Z-parameter</span><span>" + data.z.toFixed(2) + "</span></div>"
    + "<div class='row'><span class='label'>bassins</span><span>" + data.basinCount + "</span></div>"
    + "<div class='row'><span class='label'>garden-of-Eden</span><span>" + data.goePercent + "%</span></div>"
    + "<div class='row'><span class='label'>période max</span><span>" + data.maxPeriod + "</span></div>");
}

const controls = d3.select("#controls");
DATASETS.forEach((data, index) => {
  controls.append("button")
    .attr("class", "rule-btn" + (index === 1 ? " active" : ""))
    .text("règle " + data.rule + " · " + data.regime)
    .on("click", function () {
      controls.selectAll(".rule-btn").classed("active", false);
      d3.select(this).classed("active", true);
      render(index);
    });
});

render(1);

window.addEventListener("resize", () => {
  width = window.innerWidth; height = window.innerHeight;
  svg.attr("viewBox", [0, 0, width, height]);
});
</script>
</body>
</html>
"""


def main():
    datasets = []
    for rule, regime in RULES:
        graph = buildBasinGraph(computeBasinField(rule, N))
        graph["regime"] = regime
        datasets.append(graph)

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(datasets))
    outPath = Path("viz/basin_field.html")
    outPath.parent.mkdir(exist_ok=True)
    outPath.write_text(html, encoding="utf-8")

    print(f"viz écrite: {outPath}")
    for graph in datasets:
        print(f"  règle {graph['rule']:>3} ({graph['regime']:>9}) : "
              f"{len(graph['nodes'])} états, {graph['basinCount']} bassins, "
              f"Z={graph['z']}, GoE {graph['goePercent']}%")


if __name__ == "__main__":
    main()
