import os

# --- Chemins vers les données brutes (internes au repo, jamais commitées) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VICON_CSV_DIR = os.path.join(BASE_DIR, "data", "raw", "vicon_csv")
VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")

# --- Chemins internes au repo (générés automatiquement) ---
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# --- Paramètres d'acquisition ---
FS_VICON = 100.0
FS_MP_NOMINAL = 30.0

# --- Paramètres de filtrage (Butterworth passe-bas, zero-phase) ---
BUTTER_ORDER = 4
CUTOFF_HZ = 6.0

# --- Reproductibilité (test de permutation) ---
RANDOM_SEED = 42
N_PERMUTATIONS = 5000
