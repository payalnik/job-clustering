"""Self-contained HTML report for neighborhood visualizations.

One HTML file that:
  * renders every center role as its own tab,
  * draws each t-SNE point cloud on a canvas with per-group convex hulls,
  * provides hover-to-inspect tooltips and click-to-open links,
  * shows the "other similar jobs" and "farthest positions" sidebars.

All viz state is embedded as JSON — no external asset dependencies and
no fetches at view time, so the HTML opens fine from a file:// URL.
"""

from __future__ import annotations

import json
from pathlib import Path

# 17 colors so up to 16 neighbor groups never wrap.
_PALETTE = [
    "#2d5a34", "#d94f1a", "#2563eb", "#b45309", "#7c3aed",
    "#0891b2", "#be185d", "#4d7c0f", "#6d28d9", "#dc2626",
    "#0d6e6e", "#92400e", "#4338ca", "#ea580c", "#0e7490",
    "#a21caf", "#65a30d",
]


def _assign_colors(group_counts: dict[str, int], center_label: str) -> dict[str, str]:
    colors = {center_label: _PALETTE[0]}
    i = 1
    for name in group_counts:
        if name == center_label:
            continue
        colors[name] = _PALETTE[i % len(_PALETTE)]
        i += 1
    return colors


def render_html(vizzes: dict[str, dict], output_path: Path) -> None:
    """Write a single HTML page containing every center role as a tab.

    `vizzes` is {role_key: payload} where payload is what
    `neighborhood.find_neighborhood` returns. role_key is a short slug
    used in the URL hash (e.g. "ds", "pm", "swe").
    """
    enriched = {}
    for key, v in vizzes.items():
        colors = _assign_colors(v["group_counts"], v["label"])
        enriched[f"nb_{key}"] = {**v, "colors": colors}

    tab_labels = {f"nb_{k}": v["label"] for k, v in vizzes.items()}
    keys = list(vizzes.keys())

    html = (_HTML_TEMPLATE
            .replace("/*VIZ_DATA*/", json.dumps(enriched))
            .replace("/*TAB_LABELS*/", json.dumps(tab_labels))
            .replace("/*KEYS*/", json.dumps(keys)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Role Neighborhood Discovery</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', system-ui, sans-serif; background: #f5f1e8; color: #3c2a14; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
h1 { font-family: 'Fraunces', Georgia, serif; font-size: 1.6rem; margin-bottom: 4px; color: #2c1810; }
.subtitle { color: #8a7a68; font-size: 0.85rem; margin-bottom: 14px; text-align: center; max-width: 600px; }
.role-tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; justify-content: center; }
.role-tab { padding: 7px 18px; border-radius: 9999px; border: 1px solid #e8e2d6; background: #fffefa; font-size: 0.82rem; font-weight: 500; color: #5c5347; cursor: pointer; transition: all 0.2s; }
.role-tab:hover { background: #f5f0e8; }
.role-tab.active { background: #3d6b45; color: white; border-color: #3d6b45; }
#viz { width: min(900px, 95vw); height: min(650px, 75vh); background: #fffefa; border: 1px solid #e8e2d6; border-radius: 20px; box-shadow: 0 4px 20px rgba(60,50,30,0.08); position: relative; overflow: hidden; cursor: crosshair; }
canvas { width: 100%; height: 100%; }
#tooltip { position: absolute; display: none; pointer-events: none; background: #fffefa; border: 1px solid #e8e2d6; border-radius: 12px; padding: 10px 14px; font-size: 0.78rem; color: #5c5347; box-shadow: 0 6px 20px rgba(60,50,30,0.12); max-width: 280px; z-index: 10; }
#tooltip .tt-title { font-weight: 600; color: #2c1810; margin-bottom: 2px; }
#tooltip .tt-company { color: #8a7a68; }
#legend { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 12px; max-width: 900px; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.78rem; color: #5c5347; cursor: pointer; padding: 4px 12px; border-radius: 9999px; border: 1px solid #e8e2d6; background: #fffefa; transition: all 0.2s; }
.legend-item:hover { background: #f5f0e8; }
.legend-item.active { border-color: #3d6b45; background: #eef5ef; }
.legend-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.legend-count { color: #a09888; font-size: 0.65rem; }
.panel { margin-top: 16px; max-width: 900px; width: 100%; background: #fffefa; border: 1px solid #e8e2d6; border-radius: 16px; padding: 16px 20px; }
.panel h3 { font-weight: 600; font-size: 0.82rem; color: #2c1810; margin-bottom: 8px; }
.panel-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #f0ece4; font-size: 0.78rem; }
.panel-row:last-child { border-bottom: none; }
.panel-row span.company { color: #a09888; flex-shrink: 0; margin-left: 12px; }
.info { margin-top: 12px; font-size: 0.72rem; color: #a09888; text-align: center; max-width: 600px; }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
</head>
<body>
<h1>Role Neighborhood Discovery</h1>
<p class="subtitle" id="vizDesc">Full corpus with CCA-WR deconfounding — what jobs share the same responsibilities?</p>
<div class="role-tabs" id="roleTabs"></div>
<div id="viz"><canvas id="canvas"></canvas><div id="tooltip"></div></div>
<div id="legend"></div>
<div id="alsonear"></div>
<div id="farthest"></div>
<p class="info">Nearest positions by cosine similarity in a responsibility-focused embedding space. 2D layout via PCA(50) → t-SNE. Click a legend chip to solo its group. Click a dot (when solo) to open the listing.</p>

<script>
var ALL_VIZ = /*VIZ_DATA*/;
var TAB_LABELS = /*TAB_LABELS*/;
var KEYS = /*KEYS*/;

var currentViz = null, POINTS = [], COLORS = {};
var soloGroup = null, hoverGroup = null;
var canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
var tooltip = document.getElementById('tooltip'), legendEl = document.getElementById('legend');

function convexHull(pts) {
  if (pts.length < 3) return pts;
  pts = pts.slice().sort(function(a, b) { return a[0] - b[0] || a[1] - b[1]; });
  function cross(o, a, b) { return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]); }
  var lo = [], up = [];
  for (var i = 0; i < pts.length; i++) {
    while (lo.length >= 2 && cross(lo[lo.length-2], lo[lo.length-1], pts[i]) <= 0) lo.pop();
    lo.push(pts[i]);
  }
  for (var i = pts.length - 1; i >= 0; i--) {
    while (up.length >= 2 && cross(up[up.length-2], up[up.length-1], pts[i]) <= 0) up.pop();
    up.push(pts[i]);
  }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

function switchRole(key) {
  var v = ALL_VIZ[key]; if (!v) return;
  currentViz = v; POINTS = v.points; COLORS = v.colors;
  soloGroup = null; hoverGroup = null;
  history.replaceState(null, '', '#' + key.replace('nb_', ''));
  document.getElementById('vizDesc').textContent = v.description;
  document.querySelectorAll('.role-tab').forEach(function(t) {
    t.classList.toggle('active', t.dataset.key === key);
  });
  buildLegend(); buildPanels(); resize();
}

function buildLegend() {
  legendEl.innerHTML = '';
  var gc = currentViz.group_counts || {}, center = currentViz.label;
  var sorted = Object.keys(gc).sort(function(a, b) {
    if (a === center) return -1; if (b === center) return 1;
    return gc[b] - gc[a];
  });
  sorted.forEach(function(g) {
    var item = document.createElement('div');
    item.className = 'legend-item' + (soloGroup === g ? ' active' : '');
    item.innerHTML = '<span class="legend-dot" style="background:' + (COLORS[g] || '#999') + '"></span><span>' + g + '</span><span class="legend-count">' + gc[g] + '</span>';
    item.onclick = function() { soloGroup = (soloGroup === g ? null : g); buildLegend(); draw(); };
    item.onmouseenter = function() { hoverGroup = g; draw(); };
    item.onmouseleave = function() { hoverGroup = null; draw(); };
    legendEl.appendChild(item);
  });
}

function buildPanels() {
  function renderPanel(id, title, items) {
    var el = document.getElementById(id);
    if (!items || !items.length) { el.innerHTML = ''; return; }
    var h = '<div class="panel"><h3>' + title + '</h3>';
    items.forEach(function(f) {
      h += '<div class="panel-row"><span>' + f.title + '</span><span class="company">' + f.company + '</span></div>';
    });
    el.innerHTML = h + '</div>';
  }
  renderPanel('alsonear', 'Other similar jobs nearby', currentViz.also_near);
  renderPanel('farthest', 'Most different jobs in the corpus', currentViz.farthest);
}

function resize() {
  var r = canvas.parentElement.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  draw();
}

function draw() {
  var w = canvas.width / devicePixelRatio, h = canvas.height / devicePixelRatio, pad = 50;
  ctx.clearRect(0, 0, w, h);
  if (!POINTS.length) return;
  var center = currentViz.label;
  function px(v) { return pad + (v + 1) / 2 * (w - pad * 2); }
  function py(v) { return pad + (1 - (v + 1) / 2) * (h - pad * 2); }
  var groupPts = {};
  POINTS.forEach(function(p) {
    if (!groupPts[p.group]) groupPts[p.group] = [];
    groupPts[p.group].push([px(p.x), py(p.y), p]);
  });
  var hlSet = null;
  if (hoverGroup) {
    hlSet = new Set([hoverGroup]);
    if (hoverGroup !== center) hlSet.add(center);
  }
  if (soloGroup) {
    for (var gk in groupPts) {
      var pts = groupPts[gk], color = COLORS[gk] || '#999', isSolo = gk === soloGroup;
      ctx.fillStyle = color;
      ctx.globalAlpha = isSolo ? 0.75 : 0.06;
      var r = isSolo ? 4.5 : 2;
      for (var i = 0; i < pts.length; i++) {
        ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], r, 0, Math.PI * 2); ctx.fill();
      }
    }
  } else {
    for (var gk in groupPts) {
      var pts = groupPts[gk], color = COLORS[gk] || '#999';
      if (pts.length < 3) continue;
      var isHl = !hlSet || hlSet.has(gk), isFd = hlSet && !isHl;
      var cx = 0, cy = 0;
      pts.forEach(function(p) { cx += p[0]; cy += p[1]; });
      cx /= pts.length; cy /= pts.length;
      var dists = pts.map(function(p, i) {
        return { i: i, d: Math.sqrt((p[0]-cx)*(p[0]-cx) + (p[1]-cy)*(p[1]-cy)) };
      }).sort(function(a, b) { return a.d - b.d; });
      var cut = Math.floor(pts.length * 0.90);
      var core = dists.slice(0, cut).map(function(d) { return pts[d.i]; });
      if (core.length >= 3) {
        var hull = convexHull(core.map(function(p) { return [p[0], p[1]]; }));
        ctx.beginPath();
        ctx.moveTo(hull[0][0], hull[0][1]);
        for (var i = 1; i < hull.length; i++) ctx.lineTo(hull[i][0], hull[i][1]);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = isFd ? 0.01 : (isHl && hlSet ? 0.14 : 0.06);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = isHl && hlSet ? 1.5 : 1;
        ctx.globalAlpha = isFd ? 0.03 : (isHl && hlSet ? 0.40 : 0.15);
        ctx.stroke();
      }
    }
    for (var gk in groupPts) {
      var pts = groupPts[gk], color = COLORS[gk] || '#999', isCenter = gk === center;
      var isHl = !hlSet || hlSet.has(gk), isFd = hlSet && !isHl;
      ctx.fillStyle = color;
      ctx.globalAlpha = isFd ? 0.04 : (isHl && hlSet ? 0.85 : (isCenter ? 0.65 : 0.45));
      var r = isHl && hlSet ? 4 : (isCenter ? 4 : 3);
      for (var i = 0; i < pts.length; i++) {
        ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], r, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  ctx.globalAlpha = 1;
  var labels = [];
  for (var gk in groupPts) {
    var pts = groupPts[gk], cx = 0, cy = 0;
    pts.forEach(function(p) { cx += p[0]; cy += p[1]; });
    labels.push({ g: gk, x: cx / pts.length, y: cy / pts.length, w: 0, h: 18 });
  }
  ctx.font = '700 14px Inter, sans-serif';
  ctx.textAlign = 'center';
  labels.forEach(function(l) { l.w = ctx.measureText(l.g).width + 16; });
  for (var it = 0; it < 40; it++) {
    for (var i = 0; i < labels.length; i++) {
      for (var j = i + 1; j < labels.length; j++) {
        var dx = labels[j].x - labels[i].x, dy = labels[j].y - labels[i].y;
        var ox = (labels[i].w/2 + labels[j].w/2) - Math.abs(dx);
        var oy = (labels[i].h/2 + labels[j].h/2 + 6) - Math.abs(dy);
        if (ox > 0 && oy > 0) {
          var px2 = ox / 2 * (dx >= 0 ? 1 : -1);
          var py2 = oy / 2 * (dy >= 0 ? 1 : -1);
          labels[i].x -= px2 * 0.5; labels[j].x += px2 * 0.5;
          labels[i].y -= py2 * 0.5; labels[j].y += py2 * 0.5;
        }
      }
    }
  }
  labels.forEach(function(l) {
    var a = (soloGroup && l.g !== soloGroup) ? 0.15 : 1;
    ctx.globalAlpha = a;
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 4;
    ctx.lineJoin = 'round';
    ctx.strokeText(l.g, l.x, l.y + 5);
    ctx.fillStyle = COLORS[l.g] || '#444';
    ctx.fillText(l.g, l.x, l.y + 5);
  });
  ctx.globalAlpha = 1;
}

canvas.addEventListener('mousemove', function(e) {
  var r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  var w = r.width, h = r.height, pad = 50;
  function px(v) { return pad + (v + 1) / 2 * (w - pad * 2); }
  function py(v) { return pad + (1 - (v + 1) / 2) * (h - pad * 2); }
  var closest = null, minD = 15;
  POINTS.forEach(function(p) {
    if (soloGroup && p.group !== soloGroup) return;
    var x = px(p.x), y = py(p.y);
    var d = Math.sqrt((mx-x)*(mx-x) + (my-y)*(my-y));
    if (d < minD) { minD = d; closest = p; }
  });
  if (closest) {
    canvas.style.cursor = soloGroup ? 'pointer' : 'crosshair';
    tooltip.style.display = '';
    tooltip.innerHTML = '<div class="tt-title">' + closest.title + '</div><div class="tt-company">' + closest.company + '</div>';
    tooltip.style.left = Math.min(mx + 12, w - 270) + 'px';
    tooltip.style.top = Math.min(my - 10, h - 60) + 'px';
  } else {
    canvas.style.cursor = 'crosshair';
    tooltip.style.display = 'none';
  }
});
canvas.addEventListener('mouseleave', function() { tooltip.style.display = 'none'; });
canvas.addEventListener('click', function(e) {
  if (!soloGroup) return;
  var r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  var w = r.width, h = r.height, pad = 50;
  function px(v) { return pad + (v + 1) / 2 * (w - pad * 2); }
  function py(v) { return pad + (1 - (v + 1) / 2) * (h - pad * 2); }
  var closest = null, minD = 15;
  POINTS.forEach(function(p) {
    if (p.group !== soloGroup) return;
    var x = px(p.x), y = py(p.y);
    var d = Math.sqrt((mx-x)*(mx-x) + (my-y)*(my-y));
    if (d < minD) { minD = d; closest = p; }
  });
  if (closest && closest.url) window.open(closest.url, '_blank');
});

window.addEventListener('resize', resize);
window.addEventListener('hashchange', function() {
  var h = location.hash.slice(1);
  if (h && ALL_VIZ['nb_' + h]) switchRole('nb_' + h);
});

var tabsEl = document.getElementById('roleTabs');
KEYS.forEach(function(k) {
  var fullKey = 'nb_' + k;
  var tab = document.createElement('div');
  tab.className = 'role-tab';
  tab.dataset.key = fullKey;
  tab.textContent = TAB_LABELS[fullKey] || k;
  tab.onclick = function() { switchRole(fullKey); };
  tabsEl.appendChild(tab);
});

var initHash = location.hash.slice(1);
if (initHash && ALL_VIZ['nb_' + initHash]) switchRole('nb_' + initHash);
else if (KEYS.length) switchRole('nb_' + KEYS[0]);
</script>
</body></html>
"""
