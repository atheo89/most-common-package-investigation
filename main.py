import os
import time
from collections import Counter
import re
import nbformat
import plotly.express as px
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# --- Step 1: Setup Kaggle API ---
api = KaggleApi()
api.authenticate()

# --- Step 2: Prepare folders ---
os.makedirs("kaggle_notebooks", exist_ok=True)
downloaded_file = "downloaded_notebooks.txt"

# Load already downloaded notebooks
if os.path.exists(downloaded_file):
    with open(downloaded_file, "r") as f:
        downloaded = set(line.strip() for line in f.readlines())
else:
    downloaded = set()

# --- Step 3: Download notebooks ---
notebooks_list = []
pages = 1  # adjust for more notebooks
for page in range(1, pages + 1):
    notebooks_list.extend(api.kernels_list(sort_by='dateCreated', page=page, page_size=10))
print(f"Found {len(notebooks_list)} notebooks.")

for nb in notebooks_list:
    if nb.ref in downloaded:
        continue
    try:
        api.kernels_pull(nb.ref, path="kaggle_notebooks")
        print(f"Downloaded: {nb.ref}")
        downloaded.add(nb.ref)
        with open(downloaded_file, "a") as f:
            f.write(nb.ref + "\n")
        time.sleep(5)
    except Exception as e:
        print(f"Error downloading {nb.ref}: {e}")
        time.sleep(30)

# --- Step 4: Aliases, Groups, and Exclusions ---

PACKAGE_ALIASES = {
    # Scientific
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib",
    "sns": "seaborn",
    "pil": "pillow",

    # ML / AI
    "tf": "tensorflow",
    "tf2": "tensorflow",
    "tfds": "tensorflow_datasets",
    "torch": "pytorch",
    "th": "pytorch",
    "torchvision": "pytorch",
    "torchaudio": "pytorch",
    "pytorch_lightning": "pytorch",
    "lightning": "pytorch",
    "cv2": "opencv",
    "cv": "opencv",
    "sklearn": "scikit-learn",
    "xgb": "xgboost",
    "lgb": "lightgbm",
    "cat": "catboost",

    # NLP
    "transformers": "transformers",
    "hf": "transformers",
    "tokenizers": "tokenizers",
    "spacy": "spacy",
    "gensim": "gensim",
    "nltk": "nltk",
    "sentence_transformers": "transformers",

    # Visualization
    "go": "plotly",
    "px": "plotly",

    # Others / Utilities
    "tqdm.notebook": "tqdm",
    "fastai": "fastai",
    "statsmodels": "statsmodels",
    "tf_keras_vis": "tensorflow",
    "keras_hub": "tensorflow",
    "open_clip": "clip",
    "clip": "clip",
    "peft": "peft",
    "torchinfo": "torchinfo",
    "torchsummary": "torchinfo",
    "facenet_pytorch": "pytorch",
    "torch_xla": "pytorch",
    "torch_geometric": "pytorch",
    "torchio": "pytorch",
    "torchmetrics": "pytorch",
    "segmentation_models_pytorch": "pytorch",
    "pytorch_tabnet": "pytorch",
    "paddleocr": "paddleocr",
    "easyfsl": "easyfsl",
    "timm": "timm",
    "lavis": "lavis",
    "neuralop": "neuralop",
    "tensorgrad": "tensorgrad",

    # NLP / text
    "langchain": "langchain",
    "langchain_core": "langchain",
    "langchain_openai": "langchain",
    "langchain_google_genai": "langchain",
    "langchain_huggingface": "langchain",
    "langchain_experimental": "langchain",
    "langchain_text_splitters": "langchain",
    "langgraph": "langchain",
    "rank_bm25": "rank_bm25",
    "rouge_score": "rouge_score",
    "bert_score": "bert_score",
    "tiktoken": "tiktoken",
    "textblob": "textblob",
    "evaluate": "evaluate",

    # CV / image
    "skimage": "scikit-image",
    "imageio": "imageio",
    "imagecodecs": "imagecodecs",
    "tifffile": "tifffile",
    "albumentations": "albumentations",
    "ultralytics": "ultralytics",
    "supervision": "supervision",
    "roboflow": "roboflow",
    "nibabel": "nibabel",
    "pycocotools": "pycocotools",
    "rasterio": "rasterio",
    "matplotlib_venn": "matplotlib",

    # Audio / Music
    "librosa": "librosa",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
    "pretty_midi": "pretty_midi",
    "mido": "mido",
    "music21": "music21",
    "funasr": "funasr",
    "whisper": "whisper",

    # Reinforcement Learning / Gym
    "stable_baselines3": "stable_baselines3",
    "gymnasium": "gymnasium",
    "street_fighter_env": "gymnasium",
    "train_dqn": "gymnasium",

    # Distributed / Dask
    "dask_lightgbm": "dask",
    "dask_ml": "dask",
    "distributed": "dask",
    "dask": "dask",

    # Data / utils
    "polars": "polars",
    "feature_engineering": "feature_engineering",
    "category_encoders": "category_encoders",
    "optax": "optax",
    "faiss": "faiss",
    "humanize": "humanize",
    "gdown": "gdown",
    "firebase_admin": "firebase_admin",
    "pykalman": "pykalman",
    "pycountry_convert": "pycountry_convert",
    "usaddress": "usaddress",
    "pip": "pip",
}

PACKAGE_GROUPS = {
    # PyTorch ecosystem
    "pytorch": "pytorch",
    "torch": "pytorch",
    "torchvision": "pytorch",
    "torchaudio": "pytorch",
    "pytorch_lightning": "pytorch",
    "lightning": "pytorch",
    "torchio": "pytorch",
    "torchmetrics": "pytorch",
    "torch_xla": "pytorch",
    "facenet_pytorch": "pytorch",
    "segmentation_models_pytorch": "pytorch",
    "pytorch_tabnet": "pytorch",
    "torch_geometric": "pytorch",
    "timm": "pytorch",
    "lavis": "pytorch",
    "ultralytics": "pytorch",
    "supervision": "pytorch",
    "stable_baselines3": "pytorch",
    "gymnasium": "pytorch",

    # TensorFlow ecosystem
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "jax": "tensorflow",
    "tensorflow_datasets": "tensorflow",
    "tf_keras_vis": "tensorflow",
    "keras_hub": "tensorflow",
    "tensorflow_hub": "tensorflow",
    "tensorflow_addons": "tensorflow",
    "tensorflow_recommenders": "tensorflow",

    # Scikit-learn
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",

    # Boosting
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",

    # HuggingFace
    "huggingface": "huggingface",
    "transformers": "huggingface",
    "tokenizers": "huggingface",
    "datasets": "huggingface",
    "diffusers": "huggingface",
    "sentence_transformers": "huggingface",
    "huggingface_hub": "huggingface",

    # LangChain
    "langchain": "langchain",
    "langchain_core": "langchain",
    "langchain_openai": "langchain",
    "langchain_google_genai": "langchain",
    "langchain_huggingface": "langchain",
    "langchain_experimental": "langchain",
    "langchain_text_splitters": "langchain",
    "langgraph": "langchain",

    # OpenCV / image
    "opencv": "opencv",
    "skimage": "opencv",
    "imageio": "opencv",
    "tifffile": "opencv",
    "imagecodecs": "opencv",
    "albumentations": "opencv",
    "pycocotools": "opencv",
    "rasterio": "opencv",

    # Audio
    "audio": "audio",
    "librosa": "audio",
    "sounddevice": "audio",
    "soundfile": "audio",
    "pretty_midi": "audio",
    "mido": "audio",
    "music21": "audio",
    "funasr": "audio",
    "whisper": "audio",

    # Visualization
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "bokeh": "bokeh",
    "altair": "altair",

    # Data
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "fastai": "fastai",
}

# Packages to exclude (system / unrelated)
EXCLUDE_PACKAGES = {
    "os", "sys", "warnings", "gc", "time", "json", "random", "pathlib", "shutil",
    "re", "math", "typing", "" , "collections", "glob", "datetime", "csv",
    "io", "pickle", "zipfile", "uuid", "tempfile", "functools",
    "subprocess", "threading", "multiprocessing", "contextlib",
    "dataclasses", "enum", "argparse", "inspect", "types", "pprint",
    "queue", "__future__", "typing_extensions", "importlib", "hashlib",
    "base64", "string", "textwrap", "google", "kagglehub","yaml", "kaggle_evaluation", "kaggle_secrets", "learntools"
}

def normalize_package(name):
    name = name.lower()
    if name in PACKAGE_ALIASES:
        name = PACKAGE_ALIASES[name]
    if name in PACKAGE_GROUPS:
        name = PACKAGE_GROUPS[name]
    return name

# --- Step 5: Analyze notebooks ---

package_counter = Counter()
import_pattern = re.compile(r'^\s*(?:import|from)\s+([A-Za-z_][\w\.]*)')
notebooks_analyzed = 0
notebook_package_list = []

for root, dirs, files in os.walk("kaggle_notebooks"):
    for file in files:
        if file.endswith(".ipynb"):
            path = os.path.join(root, file)
            try:
                nb = nbformat.read(path, as_version=4)
                notebooks_analyzed += 1
                notebook_packages = set()

                for cell in nb.cells:
                    if cell.cell_type == "code":
                        for line in cell.source.splitlines():
                            match = import_pattern.match(line)
                            if match:
                                raw_name = match.group(1).split('.')[0]
                                normalized = normalize_package(raw_name)
                                if normalized not in EXCLUDE_PACKAGES:
                                    notebook_packages.add(normalized)

                for pkg in notebook_packages:
                    package_counter[pkg] += 1
                    notebook_package_list.append((file, pkg))

            except Exception as e:
                print(f"Error reading {file}: {e}")

# --- Step 6: Plotly chart ---
# Get top 50 packages
top_packages = package_counter.most_common(50)

if top_packages:
    packages, counts = zip(*top_packages)
    
    # Simple bar plot without color categories
    fig = px.bar(
        x=packages,
        y=counts,
        labels={'x': 'Package', 'y': 'Usage Count'},
        title=f"Top Python Packages Across {notebooks_analyzed} Kaggle public Notebooks (Unique per Notebook)"
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html("analysis_top_packages.html")
    print("Plot saved as analysis_top_packages.html")
else:
    print("No packages found in the notebooks.")


# --- Step 7: Export CSVs ---

# Aggregate CSV
agg_df = pd.DataFrame(package_counter.items(), columns=['package', 'count'])
agg_df['percent_of_notebooks'] = agg_df['count'] / notebooks_analyzed
agg_df.sort_values('count', ascending=False, inplace=True)
agg_df.to_csv("package_usage_summary.csv", index=False)
print("Aggregate package usage saved to package_usage_summary.csv")

# Notebook × Package CSV
notebook_pkg_df = pd.DataFrame(notebook_package_list, columns=['notebook', 'package'])
notebook_pkg_df.to_csv("notebook_package_matrix.csv", index=False)
print("Notebook × package matrix saved to notebook_package_matrix.csv")



