"""
experiments/poster_data_table.py

Generate a high-resolution data sources table for the poster.
GSU blue/white theme, formatted for easy reading at poster scale.

Usage:
    cd sep_detection_pipeline/
    python -m experiments.poster_data_table
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTPUT_DIR = PROJECT_ROOT / "output" / "poster_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GSU Theme
GSU_BLUE = "#0039A6"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F5"
BORDER_GRAY = "#CCCCCC"
TEXT_DARK = "#1A1A1A"

FONT = "Arial"
DPI = 300


def create_data_table():
    """Generate the data sources table as a high-res image."""
    
    # Table data
    headers = [
        "Satellite",
        "Instrument",
        "Years",
        "Flux Source"
    ]
    
    rows = [
        ["GOES-8", "EPS", "1995-2003", "p3_flux_ic (pre-computed)"],
        ["GOES-11/10", "EPS", "2003-2010", "p3_flux_ic (fallback)"],
        ["GOES-13", "EPEAD", "2010-2017", "ZPGT10E (east sensor)"],
        ["GOES-15", "EPEAD", "2018-Mar 2020", "ZPGT10E (east sensor)"],
        ["—", "—", "Apr-Oct 2020", "DATA GAP"],
        ["GOES-16", "SGPS", "Nov 2020-2025", "Derived: 13 diff. channels"],
        ["SOHO", "COSTEP/EPHIN", "1995-2025", "Derived: P8+P25+P41"],
    ]
    
    # Additional row for catalog
    catalog_row = ["GSEP Catalog", "Ground Truth", "1995-2017", "159 Flag=1 events"]
    
    n_cols = len(headers)
    n_rows = len(rows) + 1  # +1 for catalog row
    
    # Square figure: table grid is exactly square (width == 9 row heights)
    fig_size = 10.0
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Layout: 8 units wide table, row_height = 8/9 so total table height = 8 (square grid)
    table_left = 1.0
    table_width = 8.0
    n_grid_rows = 9  # header + 7 data + 1 catalog
    row_height = table_width / n_grid_rows
    catalog_y = 1.0
    table_top = catalog_y + (len(rows) + 1) * row_height
    
    col_widths = [1.8, 2.0, 2.0, 3.5]  # Proportional widths
    col_widths_normalized = [w / sum(col_widths) * table_width for w in col_widths]
    
    # Draw header row
    x_offset = table_left
    for i, (header, width) in enumerate(zip(headers, col_widths_normalized)):
        # Header box
        header_rect = Rectangle(
            (x_offset, table_top), width, row_height,
            facecolor=GSU_BLUE, edgecolor=WHITE, linewidth=1.5, zorder=2
        )
        ax.add_patch(header_rect)
        
        # Header text
        ax.text(x_offset + width / 2, table_top + row_height / 2, header,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=WHITE, fontfamily=FONT, zorder=3)
        
        x_offset += width
    
    # Draw data rows
    for row_idx, row_data in enumerate(rows):
        y = table_top - (row_idx + 1) * row_height
        
        # Special background for data gap row
        if "DATA GAP" in row_data:
            bg_color = "#FFE6E6"
        else:
            bg_color = LIGHT_GRAY if row_idx % 2 == 0 else WHITE
        
        x_offset = table_left
        for col_idx, (cell, width) in enumerate(zip(row_data, col_widths_normalized)):
            # Cell box
            cell_rect = Rectangle(
                (x_offset, y), width, row_height,
                facecolor=bg_color, edgecolor=BORDER_GRAY, linewidth=0.8, zorder=1
            )
            ax.add_patch(cell_rect)
            
            # Cell text
            fontsize = 11 if len(cell) < 25 else 10
            fontweight = "bold" if "DATA GAP" in cell else "normal"
            text_color = "#CC0000" if "DATA GAP" in cell else TEXT_DARK
            
            ax.text(x_offset + width / 2, y + row_height / 2, cell,
                    ha="center", va="center", fontsize=fontsize,
                    fontweight=fontweight, color=text_color,
                    fontfamily=FONT, zorder=2)
            
            x_offset += width
    
    # Draw catalog row (with different background)
    catalog_y = table_top - (len(rows) + 1) * row_height
    x_offset = table_left
    
    for col_idx, (cell, width) in enumerate(zip(catalog_row, col_widths_normalized)):
        cell_rect = Rectangle(
            (x_offset, catalog_y), width, row_height,
            facecolor="#FFF9E6", edgecolor=BORDER_GRAY, linewidth=0.8, zorder=1
        )
        ax.add_patch(cell_rect)
        
        fontsize = 11 if len(cell) < 25 else 10
        fontweight = "bold" if col_idx == 0 else "normal"
        
        ax.text(x_offset + width / 2, catalog_y + row_height / 2, cell,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=TEXT_DARK,
                fontfamily=FONT, zorder=2)
        
        x_offset += width
    
    # Add footnote below table (inside bottom margin of square figure)
    footnote_y = catalog_y - 0.25
    footnote = ("All instruments use 5-minute cadence. GOES-12 p3_flux_ic is broken (all -99999); " +
                "pipeline uses GOES-11 or GOES-10 as fallback.\n" +
                "Flat-spectrum partial-bin correction applied to channels crossing 10 MeV threshold " +
                "(GOES EPS P3, SOHO EPHIN P8).")
    
    ax.text(table_left, footnote_y, footnote,
            ha="left", va="top", fontsize=8, color="#555555",
            fontfamily=FONT, style="italic", wrap=True)
    
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    path = OUTPUT_DIR / "data_sources_table.png"
    # Full figure save (no tight bbox) so the PNG stays exactly square
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    create_data_table()
