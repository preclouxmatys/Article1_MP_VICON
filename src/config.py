import os

# --- Chemins vers les données brutes (externes au repo, jamais versionnées) ---
VICON_CSV_DIR = os.path.expanduser("~/Desktop/SYNCOGEST/DATA/VICON_CSV")
VIDEO_DIR = os.path.expanduser("~/Desktop/SYNCOGEST/DATA/Video")

# --- Chemins internes au repo (générés automatiquement) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# --- Paramètres d'acquisition ---
FS_VICON = 100.0       # Hz
FS_MP_NOMINAL = 30.0   # Hz (fréquence nominale caméra RGB)

# --- Paramètres de filtrage (Butterworth passe-bas, zero-phase) ---
BUTTER_ORDER = 4
CUTOFF_HZ = 6.0

# --- Reproductibilité (test de permutation) ---
RANDOM_SEED = 42
N_PERMUTATIONS = 5000
