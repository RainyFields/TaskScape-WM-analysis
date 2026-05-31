import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import numpy as np
from pathlib import Path

# ============================================================================
# GLOBAL MATPLOTLIB SETTINGS FOR ACADEMIC PUBLICATION
# ============================================================================
# Font family: Sans-serif (prefer Arial, Helvetica, DejaVu Sans)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']

# Font sizes (strict academic standards)
plt.rcParams['axes.labelsize'] = 18       # Axis labels (X/Y titles): 18pt minimum
plt.rcParams['axes.titlesize'] = 20       # Figure title
plt.rcParams['xtick.labelsize'] = 16      # Tick labels: 16pt minimum
plt.rcParams['ytick.labelsize'] = 16      # Tick labels: 16pt minimum
plt.rcParams['legend.fontsize'] = 16      # Legend text: 16pt minimum
plt.rcParams['legend.title_fontsize'] = 18  # Legend title

# Line and spine settings for visibility
plt.rcParams['axes.linewidth'] = 2.0      # Spine/border thickness
plt.rcParams['lines.linewidth'] = 4.0     # Default line width

# Tick settings
plt.rcParams['xtick.major.size'] = 8      # Major tick length
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.5   # Major tick thickness
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 4      # Minor tick length
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 1.0   # Minor tick thickness
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.direction'] = 'out'   # Ticks point outward
plt.rcParams['ytick.direction'] = 'out'

# PDF/SVG font embedding for publication
plt.rcParams['pdf.fonttype'] = 42         # TrueType fonts in PDF
plt.rcParams['ps.fonttype'] = 42          # TrueType fonts in PS
plt.rcParams['svg.fonttype'] = 'none'     # Text as text in SVG (editable)

# ============================================================================
# CONFIGURATION
# ============================================================================
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06']

# Data directory
data_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only")

# Output paths
output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/task_condition_repetition/")
output_file_png = output_dir / "task_condition_repetition_distribution.png"
output_file_pdf = output_dir / "task_condition_repetition_distribution.pdf"
output_file_svg = output_dir / "task_condition_repetition_distribution.svg"

# Output DPI for PNG (high resolution for print)
OUTPUT_DPI = 1296


def plot_multisubject_distribution():
    """
    Generate publication-quality histogram of repetition distribution 
    across multiple subjects.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use tab10 colormap for distinguishable colors
    cmap = plt.get_cmap('tab10')
    
    all_data_for_bins = []
    subject_data_store = []

    # -------------------------------------------------------------------------
    # 1. Load all data first to determine global bin sizes
    # -------------------------------------------------------------------------
    for subj in subjects:
        json_path = data_dir / f"{subj}_condition_counts.json"
        
        if not json_path.exists():
            print(f"Warning: Could not find {json_path}")
            subject_data_store.append([])
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        counts = list(data.values())
        subject_data_store.append(counts)
        all_data_for_bins.extend(counts)

    # Calculate bins based on data range
    max_rep = max(all_data_for_bins) if all_data_for_bins else 10
    bin_width = 1
    bins = np.arange(0, max_rep + 2, bin_width) - 0.5 
    
    # -------------------------------------------------------------------------
    # 2. Plot each subject with publication-quality styling
    # -------------------------------------------------------------------------
    for i, subj in enumerate(subjects):
        data = subject_data_store[i]
        if not data:
            continue
        
        # Convert numpy color tuple to hex string to avoid "Invalid RGBA" error
        raw_color = cmap(i)
        color_hex = mcolors.to_hex(raw_color)
        
        ax.hist(data, bins=bins, 
                alpha=0.4,                  # Transparency for overlapping visibility
                label=subj, 
                color=color_hex,           
                edgecolor=color_hex,       
                linewidth=5.0,              # Thick edge lines for visibility
                histtype='stepfilled')     

    # -------------------------------------------------------------------------
    # 3. Publication-Quality Styling
    # -------------------------------------------------------------------------
    # Title (optional - many journals prefer no title, remove if needed)
    ax.set_title("Distribution of Repetitions per Task Condition (All Subjects)", 
                 fontweight='bold', pad=15)
    
    # Axis labels
    ax.set_xlabel("Number of Repetitions per Condition")
    ax.set_ylabel("Number of Task Conditions")
    
    # Configure axis ticks
    ax.set_xticks(np.arange(0, max_rep + 5, step=max(1, max_rep // 20)))
    
    # Enable minor ticks for both axes
    ax.minorticks_on()
    
    # Configure tick parameters explicitly for maximum visibility
    ax.tick_params(axis='both', which='major', length=10, width=2.0, direction='out')
    ax.tick_params(axis='both', which='minor', length=5, width=1.5, direction='out')
    
    # Thicken spines (axis borders)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # Remove grid for cleaner publication look
    ax.grid(False)
    
    # Legend - positioned outside plot area to avoid overlap with data
    ax.legend(title="Subjects", 
              loc='upper left',
              bbox_to_anchor=(1.02, 1.0),   # Position outside right edge
              borderaxespad=0,
              frameon=True,
              edgecolor='black',
              framealpha=1.0)
    
    # Adjust layout to prevent clipping and accommodate external legend
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)  # Make room for external legend
    
    # -------------------------------------------------------------------------
    # 4. Save in multiple formats (PDF, SVG, PNG)
    # -------------------------------------------------------------------------
    # PNG: High resolution for print (dpi=1296)
    fig.savefig(output_file_png, dpi=OUTPUT_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # PDF: Vector format for publication (editable text)
    fig.savefig(output_file_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # SVG: Vector format for web and further editing
    fig.savefig(output_file_svg, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Figures saved to:")
    print(f"  PNG (dpi={OUTPUT_DPI}): {output_file_png}")
    print(f"  PDF: {output_file_pdf}")
    print(f"  SVG: {output_file_svg}")
    
    plt.show()


if __name__ == "__main__":
    plot_multisubject_distribution()