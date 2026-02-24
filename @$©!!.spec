# -*- mode: python ; coding: utf-8 -*-
import os
import importlib.util
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

APP_NAME = "@$©!!"
ENTRY_SCRIPT = os.path.join("app", "main.py")
ICON_PATH = os.path.join("assets", "app.ico")

def has_pkg(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

datas = []
binaries = []
hiddenimports = []

# ----------------------------
# Bundle assets/
# ----------------------------
assets_dir = os.path.abspath("assets")
if os.path.isdir(assets_dir):
    for root, _, files in os.walk(assets_dir):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(root, os.path.abspath("."))
            datas.append((src, rel))

# ----------------------------
# Bundle models/
# ----------------------------
models_dir = os.path.abspath("models")
if os.path.isdir(models_dir):
    for root, _, files in os.walk(models_dir):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(root, os.path.abspath("."))
            datas.append((src, rel))

# PIL hidden imports (safe)
hiddenimports += ["PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageFilter"]

# transformers stack
for pkg in ["transformers", "tokenizers", "safetensors", "huggingface_hub"]:
    if has_pkg(pkg):
        hiddenimports += collect_submodules(pkg)
        datas += collect_data_files(pkg, include_py_files=False)

# torch stack
if has_pkg("torch"):
    hiddenimports += collect_submodules("torch")
    binaries += collect_dynamic_libs("torch")

# NOTE: ONEDIR is strongly recommended for torch/transformers reliability
a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,   # safer for torch
    console=False,
    icon=ICON_PATH if os.path.exists(ICON_PATH) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)