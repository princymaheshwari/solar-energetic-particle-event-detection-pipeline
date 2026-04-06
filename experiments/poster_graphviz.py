"""
experiments/poster_graphviz.py

Render three Graphviz DOT diagrams for the research poster:
    1. pipeline_overview  — Main pipeline flow (left-to-right)
    2. adapter_decision   — GOES adapter fallback decision tree (top-to-bottom)
    3. detector_state     — Hysteresis detector state machine (top-to-bottom)

Outputs SVG + PNG to output/poster_figures/

Usage:
    cd sep_detection_pipeline/
    python -m experiments.poster_graphviz
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "output" / "poster_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

import graphviz


GSU_BLUE = "#0039A6"
GSU_BLUE_LIGHT = "#4A7FCC"
GSU_BLUE_PALE = "#DAE5F5"
ACCENT_ORANGE = "#CC5500"
ACCENT_ORANGE_LIGHT = "#FFF3E6"
GREEN_LIGHT = "#E6FAEF"
GREEN_BORDER = "#1B7340"
PURPLE_LIGHT = "#F0EDF5"
PURPLE_BORDER = "#4B2177"
WARM_BG = "#FFF7E6"
WARM_BORDER = "#B36B00"
GRAY_BG = "#F5F5F5"
WHITE = "#FFFFFF"


def build_pipeline_overview() -> graphviz.Digraph:
    """Main pipeline: raw data -> adapters -> flux -> detector -> fusion -> output."""

    g = graphviz.Digraph("pipeline", format="svg")
    g.attr(rankdir="LR", bgcolor="white", pad="0.4", nodesep="0.6", ranksep="0.9")
    g.attr("node", shape="rect", style="rounded,filled", fontname="Arial",
           fontsize="13", color=GSU_BLUE, penwidth="1.5")
    g.attr("edge", fontname="Arial", fontsize="10", color="#555555", arrowsize="0.8")

    # Input
    g.node("raw", "Raw Satellite Data\nGOES CSV | SOHO CDF\nGOES-16 NetCDF",
           shape="note", fillcolor=GRAY_BG, fontcolor="#111111", fontsize="12")

    # Adapter cluster
    with g.subgraph(name="cluster_adapters") as c:
        c.attr(label="Instrument Adapters", labelloc="t", style="rounded,dashed",
               color=GSU_BLUE, fontname="Arial", fontsize="12", fontcolor=GSU_BLUE)
        c.node("goes_adapt", "GOES Adapter\n\nEPS: read p3_flux_ic\nEPEAD: read ZPGT10E\nSGPS: derive from\n13 diff. channels",
               fillcolor=GSU_BLUE_PALE, fontcolor="#0A2A6E", fontsize="11")
        c.node("soho_adapt", "SOHO Adapter\n\nEPHIN differential\nflux x eff. width\nP8x15 + P25x16 + P41x12\n= >10 MeV proxy",
               fillcolor=ACCENT_ORANGE_LIGHT, fontcolor="#7A3300", fontsize="11",
               color=ACCENT_ORANGE)

    # Flux computation
    g.node("flux", "Flux Computation\n\nJ(>10 MeV) in pfu\npartial-bin correction\n(flat-spectrum approx.)",
           fillcolor=WHITE, color="#008080", fontcolor="#006060", fontsize="12")

    # Detector
    g.node("detector", "Detection Engine\n\nEntry: >= 10 pfu +\nrising gradient (3/4)\nExit: < 5 pfu for 2 hrs\nMin duration: 30 min",
           fillcolor=GREEN_LIGHT, color=GREEN_BORDER, fontcolor="#0A3D1A", fontsize="11")

    # Event extraction
    g.node("events", "Event Extraction\n& Merge\n\nmask -> intervals\nmerge gaps <= 30 min",
           fillcolor=GSU_BLUE_PALE, color=GSU_BLUE, fontcolor="#0A2A6E", fontsize="11")

    # Fusion
    g.node("fusion", "Multi-Instrument\nFusion\n\nGOES + SOHO union\nmerge overlapping\nor close (<= 30 min)",
           fillcolor=WARM_BG, color=WARM_BORDER, fontcolor="#5A3800", fontsize="11")

    # Validation
    g.node("validation", "Evaluation\n\nvs. GSEP catalog\n(1995-2017)\nPrecision / Recall / F1\nFAR / onset timing",
           fillcolor=PURPLE_LIGHT, color=PURPLE_BORDER, fontcolor="#2D1050", fontsize="11")

    # Output
    g.node("output", "Outputs\n\nEvent CSVs\nSummary metrics\nPoster figures",
           shape="note", fillcolor=GRAY_BG, fontcolor="#111111", fontsize="11")

    # Edges
    g.edge("raw", "goes_adapt")
    g.edge("raw", "soho_adapt")
    g.edge("goes_adapt", "flux")
    g.edge("soho_adapt", "flux")
    g.edge("flux", "detector")
    g.edge("detector", "events")
    g.edge("events", "fusion")
    g.edge("fusion", "validation")
    g.edge("validation", "output")

    return g


def build_adapter_decision() -> graphviz.Digraph:
    """GOES adapter fallback decision tree."""

    g = graphviz.Digraph("adapter_decision", format="svg")
    g.attr(rankdir="TB", bgcolor="white", pad="0.4", nodesep="0.5", ranksep="0.6")
    g.attr("node", shape="rect", style="rounded,filled", fontname="Arial",
           fontsize="12", penwidth="1.5")
    g.attr("edge", fontname="Arial", fontsize="10", color="#555555", arrowsize="0.8")

    g.node("start", "Input: GOES data\nfor year/month",
           shape="oval", fillcolor=GRAY_BG, color="#999999")

    g.node("d_inst", "Which instrument\nfor this year?",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11")

    g.node("eps_path", "EPS (1995-2010)",
           fillcolor=GSU_BLUE_PALE, color=GSU_BLUE, fontcolor="#0A2A6E")
    g.node("epead_path", "EPEAD (2010-2020)",
           fillcolor=GSU_BLUE_PALE, color=GSU_BLUE, fontcolor="#0A2A6E")
    g.node("sgps_path", "SGPS (2020-2025)",
           fillcolor=ACCENT_ORANGE_LIGHT, color=ACCENT_ORANGE, fontcolor="#7A3300")

    g.node("d_p3", "p3_flux_ic\npresent & valid?",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11")
    g.node("use_p3", "Use p3_flux_ic\n(pre-computed integral)",
           fillcolor=GREEN_LIGHT, color=GREEN_BORDER, fontcolor="#0A3D1A")

    g.node("d_alt", "Alternate satellite\navailable?\n(GOES-11 / GOES-10)",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11")
    g.node("use_alt", "Use alternate\np3_flux_ic\n(e.g. GOES-11)",
           fillcolor=WARM_BG, color=WARM_BORDER, fontcolor="#5A3800")

    g.node("derive", "Derive from\ndifferential P3-P7\n(flat-spectrum corr.)",
           fillcolor=WARM_BG, color=WARM_BORDER, fontcolor="#5A3800")

    g.node("use_epead", "Read ZPGT10E\n(east sensor integral)\nfallback: ZPGT10W",
           fillcolor=GREEN_LIGHT, color=GREEN_BORDER, fontcolor="#0A3D1A")

    g.node("use_sgps", "Derive from 13 diff.\nchannels (>= 10 MeV)\nsum(flux x dE)",
           fillcolor=GREEN_LIGHT, color=GREEN_BORDER, fontcolor="#0A3D1A")

    g.node("done", "Return >10 MeV\nflux in pfu",
           shape="oval", fillcolor=GRAY_BG, color="#999999")

    g.node("gap", "2020 Apr-Oct:\nNo GOES data\n(return empty)",
           fillcolor="#FFF0F0", color="#CC0000", fontcolor="#990000", fontsize="11")

    # Edges
    g.edge("start", "d_inst")
    g.edge("d_inst", "eps_path", label="1995-2010")
    g.edge("d_inst", "epead_path", label="2010-2020")
    g.edge("d_inst", "sgps_path", label="2020-2025")
    g.edge("d_inst", "gap", label="2020\nApr-Oct", style="dashed", color="#CC0000")

    g.edge("eps_path", "d_p3")
    g.edge("d_p3", "use_p3", label="yes")
    g.edge("d_p3", "d_alt", label="no\n(GOES-12\nall -99999)")
    g.edge("d_alt", "use_alt", label="yes")
    g.edge("d_alt", "derive", label="no")

    g.edge("epead_path", "use_epead")
    g.edge("sgps_path", "use_sgps")

    g.edge("use_p3", "done")
    g.edge("use_alt", "done")
    g.edge("derive", "done")
    g.edge("use_epead", "done")
    g.edge("use_sgps", "done")

    return g


def build_detector_state() -> graphviz.Digraph:
    """Hysteresis detector state machine."""

    g = graphviz.Digraph("detector_state", format="svg")
    g.attr(rankdir="TB", bgcolor="white", pad="0.4", nodesep="0.5", ranksep="0.7")
    g.attr("node", shape="rect", style="rounded,filled", fontname="Arial",
           fontsize="12", penwidth="1.5")
    g.attr("edge", fontname="Arial", fontsize="10", color="#555555", arrowsize="0.8")

    g.node("idle", "IDLE\nNo active event",
           shape="oval", fillcolor=GRAY_BG, color="#999999", fontsize="13")

    g.node("check_entry", "flux >= 10 pfu\nAND\n3/4 gradients positive?",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11",
           width="2.8")

    g.node("in_event", "IN EVENT\nRecording detection",
           fillcolor=GREEN_LIGHT, color=GREEN_BORDER, fontcolor="#0A3D1A",
           fontsize="13")

    g.node("check_exit", "flux < 5 pfu?",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11")

    g.node("quiet_timer", "Quiet Timer\nCounting consecutive\npoints < 5 pfu\n(need 24 = 2 hours)",
           fillcolor=WARM_BG, color=WARM_BORDER, fontcolor="#5A3800")

    g.node("check_quiet", "24 consecutive\npoints < 5 pfu?",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11")

    g.node("end_event", "END EVENT\nBackdate end to first\npoint below 5 pfu",
           fillcolor="#FFF0F5", color="#990050", fontcolor="#660033", fontsize="12")

    g.node("check_dur", "Duration\n>= 30 min?",
           shape="diamond", fillcolor=GSU_BLUE, fontcolor="white", fontsize="11")

    g.node("emit", "EMIT EVENT\nRecord start/end/duration",
           fillcolor=GREEN_LIGHT, color=GREEN_BORDER, fontcolor="#0A3D1A",
           fontsize="12")

    g.node("discard", "DISCARD\n(noise / artifact)",
           fillcolor="#FFF0F0", color="#CC0000", fontcolor="#990000", fontsize="11")

    # Edges
    g.edge("idle", "check_entry", label="next\ntimestamp")
    g.edge("check_entry", "in_event", label="YES\n(event starts)")
    g.edge("check_entry", "idle", label="NO", style="dashed", constraint="false")

    g.edge("in_event", "check_exit", label="next\ntimestamp")
    g.edge("check_exit", "quiet_timer", label="YES")
    g.edge("check_exit", "in_event", label="NO\n(event\ncontinues)", style="dashed", constraint="false")

    g.edge("quiet_timer", "check_quiet")
    g.edge("check_quiet", "end_event", label="YES\n(2 hrs elapsed)")
    g.edge("check_quiet", "in_event", label="NO\nflux >= 5 pfu\n(timer resets)", style="dashed")

    g.edge("end_event", "check_dur")
    g.edge("check_dur", "emit", label="YES")
    g.edge("check_dur", "discard", label="NO\n(< 30 min)")

    g.edge("emit", "idle", style="dotted")
    g.edge("discard", "idle", style="dotted")

    return g


def render_graph(g: graphviz.Digraph, name: str):
    """Render a graph to SVG and PNG."""
    base = str(OUTPUT_DIR / name)

    g.format = "svg"
    svg_path = g.render(filename=base, cleanup=True)
    print(f"  SVG: {svg_path}")

    g.format = "png"
    g.attr(dpi="300")
    png_path = g.render(filename=base, cleanup=True)
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    print("Rendering Graphviz diagrams...\n")

    print("[1/3] Pipeline Overview")
    render_graph(build_pipeline_overview(), "pipeline_overview")

    print("\n[2/3] Adapter Decision Tree")
    render_graph(build_adapter_decision(), "adapter_decision")

    print("\n[3/3] Detector State Machine")
    render_graph(build_detector_state(), "detector_state")

    print(f"\nAll diagrams saved to: {OUTPUT_DIR}")
