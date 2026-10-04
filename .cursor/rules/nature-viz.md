# 🔬 Nature Scientific Data: Figure Styling Standards

## 1. Typography & Hierarchy
Precision in text is non-negotiable. Nature requires specific font types to ensure readability across print and digital platforms.

| Element | Size (pt) | Weight | Note |
| :--- | :--- | :--- | :--- |
| **Panel Labels** | 8pt | **Bold** | Lowercase (**a**, **b**, **c**) |
| **Figure Titles** | 8pt | Regular | Usually only used in internal drafts. |
| **Axis Labels** | 7pt | Regular | Use sentence case. Include units in (). |
| **Tick/Legend** | 6pt | Regular | Smallest standard legible text. |
| **Minimum** | 5pt | Regular | Only for dense data points or annotations. |

---

## 2. Technical Specifications
### Geometry & Spacing
* **Dimensions:** Standard widths are **89 mm** (single column) or **183 mm** (double column).
* **Spines:** 0.5pt width. Keep them thin; the data must be the primary focus.
* **Ticks:** Length: 3pt. Direction: **Inward**. Width: 0.5pt.
* **Gridlines:** Generally avoided. If essential, use 0.3pt width with alpha 0.3.

### Color & Accessibility
* **Color-Blind Safety:** Use palettes like `viridis`, `magma`, or `Wong` for general plots.
* **Heatmaps:** Use `turbo` colormap for all heatmap/matrix visualizations.
* **Combinations:** Never use Red-Green combinations. 
* **Background:** Always use pure white (`#FFFFFF`).

---

## 3. Implementation Snippet (Python)

This snippet handles the conversion from millimeters to inches and ensures fonts are embedded correctly for PDF export.

```python
import matplotlib.pyplot as plt

def set_nature_rcparams():
    """Sets global rcParams for Nature-style plotting."""
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,      # Embeds fonts in PDF (Critical for journals)
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.dpi': 600,
        'savefig.transparent': True,
        'savefig.bbox': 'tight'
    }
    plt.rcParams.update(params)

def create_nature_fig(width_mm=89, height_mm=60):
    """Initializes a figure with specific physical dimensions."""
    set_nature_rcparams()
    # Convert mm to inches
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

def save_publication_fig(fig, filename):
    """Saves high-res Raster (PNG) and Vector (PDF) copies."""
    # 1. Vector (Publication Standard)
    fig.savefig(f"{filename}.pdf", format='pdf')
    # 2. Raster (Preview/Presentation)
    fig.savefig(f"{filename}.png", format='png', dpi=600, facecolor='white')
    print(f"Exported {filename} (.pdf, .png)")