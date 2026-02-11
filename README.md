# Mapstitcher fork with GUI, primarily CUDA Windows

This is a fork.

## Installation

### Prerequisites
- Python 3.13
- Git
- OpenJPEG
- For GPU (LoFTR/RAFT): NVIDIA GPU + up-to-date driver
- For CPU-only usage: use `--matching-algorithm sift`

> Note: `requirements.txt` installs PyTorch nightly CUDA wheels (`cu130`) via `--index-url`.
> If CUDA wheels cannot be downloaded/installed, use CPU PyTorch and/or switch to SIFT.

---
### Clone this repo

```powershell
git clone https://github.com/bezverec/mapstitcher.git
```

### Create a virtual environment & install dependencies

#### Windows (PowerShell / Terminal)
```powershell
cd C:\path\to\mapstitcher
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

#### Linux/macOS
```bash
cd /path/to/mapstitcher
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

---

## Run (CLI)

### Option A: Input folder (`--path`)
The input folder can contain any number of images named like:
`img_<row>_<col>.<ext>` (e.g. `img_0_0.jpg`, `img_0_1.jpg`, `img_1_0.jpg`, ...)

```bash
python image_stitch_batch.py --path ./test_data --output result.jp2
```

### Option B: Config file (`--list`)
```bash
python image_stitch_batch.py --list list.txt --output result.jp2
```

---

## Run (Tkinter GUI)

### Start the GUI
Activate the same venv you installed dependencies into, then run the GUI script.

#### Windows
```powershell
.\venv\Scripts\Activate.ps1
python mapstitcher_gui.py
```

#### Linux/macOS
```bash
source venv/bin/activate
python3 mapstitcher_gui.py
```

---

## JP2 output (OpenJPEG)

MapStitcher uses `opj_compress` (OpenJPEG tools) when the output filename ends with `.jp2`.

### Linux (Debian/Ubuntu)
```bash
sudo apt install libopenjp2-tools
```

### Windows
Either:
1) Put `opj_compress.exe` next to the GUI in:
   `./openjpeg/bin/opj_compress.exe`
   (relative to the GUI script folder), **or**
2) Add the folder containing `opj_compress.exe` to your `PATH`.

---

## Troubleshooting

### `Torch not compiled with CUDA enabled`
You installed CPU-only torch. Reinstall CUDA-enabled torch (cu128) or run on CPU with SIFT:

```bash
python image_stitch_batch.py --path ./test_data --matching-algorithm sift
```

### `.jp2` fails / `opj_compress` not found
Install OpenJPEG tools or provide `opj_compress` via PATH (Windows: `./openjpeg/bin/` next to the GUI is supported).

### Download / wheel issues
Upgrade pip and retry:
```bash
python -m pip install -U pip
```
