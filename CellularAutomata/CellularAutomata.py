import os
import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageDraw

# Setup -----------------------------------------------------
GRID_WIDTH = 100
GRID_HEIGHT = 100

BASE_VIEWPORT_CELLS = 50
MIN_ZOOM = 1
MAX_ZOOM = 5
DEFAULT_RANDOM_LIVE_PERCENTAGE = 50

#PATTERNS_DIR = "patterns"
PATTERNS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "patterns"
)

def load_patterns_from_csv():
    patterns = {}
    for filename in os.listdir(PATTERNS_DIR):
        file_path = os.path.join(PATTERNS_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        name, _ext = os.path.splitext(filename)
        df = pd.read_csv(file_path, header=None)
        patterns[name] = df.to_numpy()
    return patterns

patterns = load_patterns_from_csv()

PRESET_RULES = {
    "Custom": ([], []),
    "Conway's Life (B3/S23)": ([3], [2, 3]),
    "34 Life (B34/S34)": ([3, 4], [3, 4]),
    "HighLife (B36/S23)": ([3, 6], [2, 3]),
    "Day & Night (B3678/S34678)": ([3, 6, 7, 8], [3, 4, 6, 7, 8]),
    "Seeds (B2/S)": ([2], []),
    "Replicator (B1357/S1357)": ([1, 3, 5, 7], [1, 3, 5, 7]),
    "Amoeba (B357/S1358)": ([3, 5, 7], [1, 3, 5, 8]),
    "Diamoeba (B35678/S5678)": ([3, 5, 6, 7 ,8], [5, 6, 7, 8]),
    "2x2 (B36/S125)": ([3, 6,], [1, 2, 5])
}

COLOUR_MAP = {
    "Black": ((0, 0, 0), (100, 100, 100)),
    "Red": ((255, 0, 0), (255, 120, 120)),
    "Green": ((0, 180, 0), (150, 255, 150)),
    "Blue": ((0, 0, 255), (150, 150, 255)),
    "Purple": ((128, 0, 128), (200, 150, 200)),
    "Orange": ((255, 140, 0), (255, 200, 120))
}



# Functions ---------------------------------------------------------
def make_grid(width, height, live_percentage=5):
    total_cells = width * height
    live_cells = int(live_percentage * total_cells / 100)
    grid = np.zeros((height, width), dtype=int)
    live_positions = np.random.choice(total_cells, live_cells, replace=False)
    grid[np.unravel_index(live_positions, grid.shape)] = 1 
    return grid

def update_grid_two_state(grid, born_rule, survive_rule, totalistic=False):
    new_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            live_neighbours = sum([
                grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] == 1,
                grid[(i-1)%grid.shape[0], j] == 1,
                grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] == 1,
                grid[i, (j-1)%grid.shape[1]] == 1,
                grid[i, (j+1)%grid.shape[1]] == 1,
                grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] == 1,
                grid[(i+1)%grid.shape[0], j] == 1,
                grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]] == 1,
            ])
            if totalistic:
                live_neighbours += (grid[i, j] == 1)

            if grid[i, j] == 1:
                if live_neighbours in survive_rule:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 0
            else:
                if live_neighbours in born_rule:
                    new_grid[i, j] = 1
    return new_grid

def update_grid_three_state(grid, born_rule, survive_rule, totalistic=False):
    new_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            live_neighbours = sum([
                grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] == 1,
                grid[(i-1)%grid.shape[0], j] == 1,
                grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] == 1,
                grid[i, (j-1)%grid.shape[1]] == 1,
                grid[i, (j+1)%grid.shape[1]] == 1,
                grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] == 1,
                grid[(i+1)%grid.shape[0], j] == 1,
                grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]] == 1,
            ])
            if totalistic:
                live_neighbours += (grid[i, j] == 1)

            if grid[i, j] == 1:
                if live_neighbours in survive_rule:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 2
            elif new_grid[i, j] == 2:   
                 new_grid[i, j] = 0        
            else:
                if live_neighbours in born_rule:
                    new_grid[i, j] = 1
    return new_grid

def grid_to_image(grid, cell_size=10, draw_grid=True):
    height, width = grid.shape
    img = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(img)

    alive_colour, dying_colour = COLOUR_MAP.get(
        st.session_state.get("colour_choice", "Black"), ((0, 0, 0), (128, 128, 128))
    )

    for i in range(height):
        for j in range(width):
            top_left = (j * cell_size, i * cell_size)
            bottom_right = ((j + 1) * cell_size - 1, (i + 1) * cell_size - 1)
            value = grid[i, j]
            if value == 1:
                colour = alive_colour
            elif value == 2:
                colour = dying_colour
            else:
                colour = (255, 255, 255)
            draw.rectangle([top_left, bottom_right], fill=colour)
            if draw_grid:
                draw.rectangle([top_left, bottom_right], outline="gray")
    return img



def get_viewport(grid, zoom_level):
    viewport_size = max(5, BASE_VIEWPORT_CELLS // zoom_level)
    center_row = GRID_HEIGHT // 2
    center_col = GRID_WIDTH // 2
    half_size = viewport_size // 2
    row_start = max(0, center_row - half_size)
    row_end = min(GRID_HEIGHT, center_row + half_size)
    col_start = max(0, center_col - half_size)
    col_end = min(GRID_WIDTH, center_col + half_size)
    return grid[row_start:row_end, col_start:col_end]

# Streamlit Setup -----------------------------------------------==================
st.set_page_config(page_title="Cellular Automaton", layout="wide")
st.title("Cellular Automaton")

# Session State Manager -----------------------------------------------------------
def init_state():
    if "grid" not in st.session_state:
        st.session_state.grid = make_grid(GRID_WIDTH, GRID_HEIGHT, DEFAULT_RANDOM_LIVE_PERCENTAGE)
        st.session_state.running = False
        st.session_state.born_rule = [3]
        st.session_state.survive_rule = [2, 3]
        st.session_state.iteration = 0

init_state()

# Sidebar ---------------------------------------------------------------------------------
st.sidebar.header("Neighborhood Rules")
st.session_state.born_rule = st.sidebar.multiselect("Born (dead→live)", list(range(10)), default=st.session_state.born_rule)
st.session_state.survive_rule = st.sidebar.multiselect("Survive (stay live)", list(range(10)), default=st.session_state.survive_rule)

st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("Automaton Mode", ["2-state (Dead/Alive)", "3-state (Dead/Alive/Dying)"])

st.sidebar.header("Patterns")
pattern_names = list(patterns.keys())
selected_pattern = st.sidebar.selectbox("Choose Pattern", pattern_names)
if st.sidebar.button("Load Pattern"):
    st.session_state.grid = patterns[selected_pattern]
    st.session_state.running = False
    st.session_state.iteration = 0

totalistic = st.sidebar.checkbox("Use Totalistic (include self)", value=False)
zoom_level = st.sidebar.slider("Zoom Level", MIN_ZOOM, MAX_ZOOM, 1)

# Main Controls ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
 if c1.button("Next Step"):
    if mode == "2-state (Dead/Alive)":
        st.session_state.grid = update_grid_two_state(st.session_state.grid, st.session_state.born_rule, st.session_state.survive_rule, totalistic)
    else:
        st.session_state.grid = update_grid_three_state(st.session_state.grid, st.session_state.born_rule, st.session_state.survive_rule, totalistic)
    st.session_state.iteration += 1


with c2:
 if c2.button("Start" if not st.session_state.running else "Stop"):
    st.session_state.running = not st.session_state.running

with c3:
 if c3.button("Random grid"):
    st.session_state.grid = make_grid(GRID_WIDTH, GRID_HEIGHT, DEFAULT_RANDOM_LIVE_PERCENTAGE)
    st.session_state.running = False
    st.session_state.iteration = 0

with c4:
 colour_choice = st.selectbox("Colour", list(COLOUR_MAP.keys()))
 st.session_state.colour_choice = colour_choice


fps = st.slider("Frames per Second (FPS)", 1, 60, 5, step=1)
delay_ms = int(1000 / fps)

preset = st.selectbox("Preset Rules", list(PRESET_RULES.keys()))
if preset != "Custom":
    born, survive = PRESET_RULES[preset]   
    st.session_state.born_rule = born
    st.session_state.survive_rule = survive

# Display -------------------------------------------------------------------------
visible_grid = get_viewport(st.session_state.grid, zoom_level)
cell_size = 10 * zoom_level
img = grid_to_image(visible_grid, cell_size=cell_size, draw_grid=True)
col_left, col_mid, col_right = st.columns([1, 6, 1])
with col_mid:
    st.markdown(f"### t = {st.session_state.iteration}")
    st.image(img, use_container_width=False, caption="Grid View")

# Auto-Reruns-------------------------------------------------------------------
if st.session_state.running:

    
    if mode == "2-state (Dead/Alive)" :
          st.session_state.grid = update_grid_two_state(st.session_state.grid, st.session_state.born_rule, st.session_state.survive_rule, totalistic)

    else:
        st.session_state.grid = update_grid_three_state(st.session_state.grid, st.session_state.born_rule, st.session_state.survive_rule, totalistic)

    st.session_state.iteration +=1
    time.sleep(delay_ms /1000)
    st.rerun()
    
