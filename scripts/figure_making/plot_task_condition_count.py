import json
import matplotlib.pyplot as plt
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
plt.rcParams['lines.linewidth'] = 5.0     # Default line width
plt.rcParams['lines.markersize'] = 14     # Default marker size

# Tick settings
plt.rcParams['xtick.major.size'] = 10     # Major tick length
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2.0   # Major tick thickness
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.minor.size'] = 5      # Minor tick length
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5   # Minor tick thickness
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.direction'] = 'out'   # Ticks point outward
plt.rcParams['ytick.direction'] = 'out'

# PDF/SVG font embedding for publication
plt.rcParams['pdf.fonttype'] = 42         # TrueType fonts in PDF
plt.rcParams['ps.fonttype'] = 42          # TrueType fonts in PS
plt.rcParams['svg.fonttype'] = 'none'     # Text as text in SVG (editable)

# ============================================================================
# CONFIGURATION
# ============================================================================
data_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only")
output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/task_condition_repetition")

# Output DPI for PNG (high resolution for print)
OUTPUT_DPI = 1296


def plot_task_condition_count():
    """
    Generate publication-quality plot of task conditions with >5 repetitions
    per subject.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. COLLECT DATA
    # -------------------------------------------------------------------------
    subject_data = []

    # Find all JSON count files
    json_files = sorted(data_dir.glob("*_condition_counts.json"))

    if not json_files:
        print(f"No JSON count files found in {data_dir}")
        return

    for json_file in json_files:
        # Extract subject ID
        subj_id = json_file.name.split('_')[0] 
        
        with open(json_file, 'r') as f:
            counts = json.load(f)
            
        # LOGIC: Count conditions with > 5 repetitions
        num_conditions_over_5 = sum(1 for count in counts.values() if count > 5)
        
        subject_data.append((subj_id, num_conditions_over_5))

    # Sort by subject ID
    subject_data.sort(key=lambda x: x[0]) 

    subjects = [x[0] for x in subject_data]
    values = [x[1] for x in subject_data]

    # -------------------------------------------------------------------------
    # 2. CREATE FIGURE WITH PUBLICATION-QUALITY STYLING
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create the line plot with THICK lines and large markers
    ax.plot(subjects, values, 
            marker='o', 
            linestyle='-', 
            color='#1f77b4',      # Standard blue
            linewidth=5.0,        # Thick line for visibility
            markersize=14,        # Large markers
            markeredgewidth=2.0,  # Thick marker edge
            markeredgecolor='#1f77b4',
            markerfacecolor='white')
    
    # -------------------------------------------------------------------------
    # 3. PUBLICATION-QUALITY STYLING
    # -------------------------------------------------------------------------
    # Title
    ax.set_title("Total Task Conditions with > 5 Repetitions per Subject", 
                 fontweight='bold', pad=20)
    
    # Axis labels
    ax.set_xlabel("Subject", labelpad=15)
    ax.set_ylabel("Number of Conditions (>5 reps)", labelpad=15)
    
    # Set Y-axis range
    ax.set_ylim(200, 500)
    
    # Rotate x-tick labels
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    
    # Enable minor ticks
    ax.minorticks_on()
    
    # Configure tick parameters explicitly for maximum visibility
    ax.tick_params(axis='both', which='major', length=10, width=2.0, direction='out')
    ax.tick_params(axis='both', which='minor', length=5, width=1.5, direction='out')
    ax.tick_params(axis='x', which='minor', bottom=False)  # No minor ticks for categorical x-axis
    
    # Thicken spines (axis borders)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # Grid styling - subtle but visible
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_axisbelow(True)  # Grid behind data
    
    # Add value labels on top of points
    for x, y in zip(subjects, values):
        ax.text(x, y + 12, str(y), ha='center', fontsize=14, fontweight='bold')

    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # 4. SAVE IN MULTIPLE FORMATS (PNG, PDF, SVG)
    # -------------------------------------------------------------------------
    base_filename = "summary_count_task_conditions_over_all_subjects"
    
    # PNG: High resolution for print (dpi=1296)
    fig.savefig(output_dir / f"{base_filename}.png", dpi=OUTPUT_DPI, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved PNG (dpi={OUTPUT_DPI}) to {output_dir / f'{base_filename}.png'}")
    
    # PDF: Vector format for publication
    fig.savefig(output_dir / f"{base_filename}.pdf", format='pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved PDF to {output_dir / f'{base_filename}.pdf'}")
    
    # SVG: Vector format for web and further editing
    fig.savefig(output_dir / f"{base_filename}.svg", format='svg',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved SVG to {output_dir / f'{base_filename}.svg'}")
    
    plt.show()


if __name__ == "__main__":
    plot_task_condition_count()

