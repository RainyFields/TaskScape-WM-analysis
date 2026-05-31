import pickle
import plotly.graph_objects as go
import numpy as np

subj = "sub-06"

# ---------------------------------------------------------
# 1. COLOR & SATURATION PALETTE
# ---------------------------------------------------------

TASK_SPECIFIC_COLORS = {
    # --- GROUP 1: NBACK (Yellow Family) ---
    'nback_loc': "#F1C40F",  # Strong Gold
    'nback_ctg': "#F7DC6F",  # Medium Yellow
    'nback_obj': "#FCF3CF",  # Pale Yellow

    # --- GROUP 2: CTXDM (Teal Family) ---
    'ctxdm_col': "#16A085",  # Dark Teal
    'ctxdm_lco': "#76D7C4",  # Light Teal

    # --- GROUP 3: INTERDMS (Purple Family) ---
    'interdms_loc_ABAB': "#6C3483", # Deepest Purple
    'interdms_loc_ABBA': "#8E44AD", # Deep Purple
    'interdms_obj_ABAB': "#A569BD", # Medium Purple
    'interdms_obj_ABBA': "#BB8FCE", # Soft Purple
    'interdms_ctg_ABAB': "#D2B4DE", # Pale Lavender
    'interdms_ctg_ABBA': "#E8DAEF", # Very Pale Lavender
}

DEFAULT_COLOR = "#888888"

def hex_to_rgba(hex_color, alpha=0.6):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def get_task_color(task_name, alpha=0.6):
    hex_code = TASK_SPECIFIC_COLORS.get(task_name, DEFAULT_COLOR)
    return hex_to_rgba(hex_code, alpha)

# ---------------------------------------------------------
# 2. DATA LOADING & PARSING
# ---------------------------------------------------------

pkl_path = f"/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/{subj}_task_condition_betas.pkl"

with open(pkl_path, "rb") as f:
    pkl_data = pickle.load(f)

ALL_KNOWN_TASKS = [
    'nback_loc', 'nback_ctg', 'nback_obj',
    'ctxdm_col', 'ctxdm_lco',
    'interdms_loc_ABAB', 'interdms_loc_ABBA',
    'interdms_obj_ABAB', 'interdms_obj_ABBA',
    'interdms_ctg_ABAB', 'interdms_ctg_ABBA',
]

def parse_key_strict(key):
    sorted_known = sorted(ALL_KNOWN_TASKS, key=len, reverse=True)
    for task in sorted_known:
        if key.startswith(task):
            suffix = key[len(task):]
            if suffix.startswith('_'): suffix = suffix[1:]
            return task, suffix
    return "Unknown", key

def get_task_features(task_name):
    tn = task_name.lower()
    if "ctxdm" in tn: return ["Location", "Object", "Category"]
    if "loc" in tn: return ["Location"]
    elif "obj" in tn: return ["Object"]
    elif "ctg" in tn or "cat" in tn: return ["Category"]
    return ["Unknown Feature"]

# ---------------------------------------------------------
# 3. SORTING LOGIC
# ---------------------------------------------------------

unique_tasks = set()
unique_features = set()
unique_conditions_map = {} 

for key in pkl_data.keys():
    task_name, condition_suffix = parse_key_strict(key)
    unique_tasks.add(task_name)
    
    feats = get_task_features(task_name)
    for f in feats:
        unique_features.add(f)
    
    # Map condition to (PrimaryFeature, TaskName) for sorting
    primary_feature = feats[0] 
    unique_conditions_map[condition_suffix] = (primary_feature, task_name)

# 1. Tasks: Custom Order
sorted_tasks = [t for t in ALL_KNOWN_TASKS if t in unique_tasks]
remaining = [t for t in unique_tasks if t not in sorted_tasks]
sorted_tasks.extend(sorted(remaining))

# 2. Features: Alphabetical
sorted_features = sorted(list(unique_features))

# 3. Conditions: Sorted by Feature -> Task
sorted_condition_keys = sorted(
    unique_conditions_map.keys(),
    key=lambda x: (unique_conditions_map[x][0], unique_conditions_map[x][1], x)
)
sorted_conditions = sorted_condition_keys

# ---------------------------------------------------------
# 4. PREPARE NODES
# ---------------------------------------------------------

all_labels_map = {}
final_labels_list = []
final_x_list = []
final_colors_list = []

def format_label(text, size, bold=False):
    weight = "bold" if bold else "normal"
    return f"<span style='font-size: {size}px; font-weight: {weight}'>{text}</span>"

# Add TASKS
for t in sorted_tasks:
    all_labels_map[t] = len(final_labels_list)
    final_labels_list.append(format_label(t, 24, bold=True))
    final_x_list.append(0.01)
    final_colors_list.append(get_task_color(t, alpha=1.0))

# Add FEATURES
for f in sorted_features:
    all_labels_map[f] = len(final_labels_list)
    final_labels_list.append(format_label(f, 24, bold=True))
    final_x_list.append(0.35)
    final_colors_list.append("#555555")

# Add CONDITIONS
total_conditions = len(sorted_conditions)
# Reduced frequency slightly (divide by 15 instead of 20) to make room for bigger font
label_step = max(1, total_conditions // 15) 

for i, c in enumerate(sorted_conditions):
    all_labels_map[c] = len(final_labels_list)
    
    if i % label_step == 0:
        # --- CHANGED: Font Size increased to 16px, Bold = True ---
        final_labels_list.append(format_label(c, 16, bold=True))
    else:
        final_labels_list.append("") 
        
    final_x_list.append(0.99)
    final_colors_list.append("#555555")

# ---------------------------------------------------------
# 5. BUILD LINKS
# ---------------------------------------------------------

sources = []
targets = []
values = []
link_colors = []

for key, beta_list in pkl_data.items():
    count = len(beta_list)
    task_name, condition_suffix = parse_key_strict(key)
    relevant_features = get_task_features(task_name)
    
    flow_color = get_task_color(task_name, alpha=0.6)
    
    if task_name not in all_labels_map or condition_suffix not in all_labels_map:
        continue
    
    idx_task = all_labels_map[task_name]
    idx_cond = all_labels_map[condition_suffix]
    
    for feat in relevant_features:
        if feat not in all_labels_map: continue
        idx_feat = all_labels_map[feat]
        
        sources.append(idx_task)
        targets.append(idx_feat)
        values.append(count)
        link_colors.append(flow_color)
        
        sources.append(idx_feat)
        targets.append(idx_cond)
        values.append(count)
        link_colors.append(flow_color)

# ---------------------------------------------------------
# 6. PLOT
# ---------------------------------------------------------

fig = go.Figure(data=[go.Sankey(
    arrangement = "snap",
    textfont = dict(family="Arial", color="black"), 
    node = dict(
      pad = 2, 
      thickness = 15, 
      line = dict(color = "white", width = 0.5),
      label = final_labels_list,
      x = final_x_list,
      y = [None] * len(final_labels_list),
      color = final_colors_list,
      hovertemplate='%{label}<extra></extra>'
    ),
    link = dict(
      source = sources,
      target = targets,
      value = values,
      color = link_colors,
      hovertemplate='%{source.label} → %{target.label}<extra></extra>'
  ))])

fig.update_layout(
    title_text=" ",
    font_size=12,
    width=1600,
    height=1600, 
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=50, t=50, b=50)
)

try:
    fig.write_image(f"/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/{subj}_sankey_diagram.pdf") 
    fig.write_image(f"/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/{subj}_sankey_diagram.png") 
    print("Saved sankey_diagram.png and .pdf")
except Exception as e:
    print(f"Image export failed: {e}")
    fig.write_html(f"/project/def-pbellec/xuan/fmri_dataset_project/scripts/figure_making/figures/{subj}_sankey_diagram.html")
    print("Saved sankey_diagram.html")