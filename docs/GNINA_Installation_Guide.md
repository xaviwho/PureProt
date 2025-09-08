# GNINA Installation Guide for Windows

GNINA is a deep learning-enhanced molecular docking framework that provides superior accuracy compared to traditional docking methods. This guide covers installation options for Windows users.

## Overview

GNINA uses convolutional neural networks (CNNs) for scoring protein-ligand interactions, making it more accurate than traditional scoring functions. It's particularly useful for virtual screening and drug discovery applications.

## Installation Options

### Option 1: Pre-built Binary (Recommended for Windows)

The easiest way to use GNINA on Windows is through the pre-built Linux binary with WSL2.

#### Prerequisites
1. **Windows Subsystem for Linux 2 (WSL2)**
   ```powershell
   # Run in PowerShell as Administrator
   wsl --install
   # Restart your computer when prompted
   ```

2. **Install Ubuntu 22.04 from Microsoft Store**
   - Open Microsoft Store
   - Search for "Ubuntu 22.04"
   - Install and launch

#### Step-by-Step Installation

1. **Download the Latest GNINA Binary**
   ```bash
   # In WSL2 Ubuntu terminal
   cd ~
   wget https://github.com/gnina/gnina/releases/latest/download/gnina
   chmod +x gnina
   sudo mv gnina /usr/local/bin/
   ```

2. **Verify Installation**
   ```bash
   gnina --help
   ```

### Option 2: Docker (Alternative)

If you prefer Docker, GNINA provides pre-built Docker images:

```bash
# Pull the latest GNINA Docker image
docker pull gnina/gnina:latest

# Run GNINA in Docker
docker run -v $(pwd):/data gnina/gnina:latest gnina --help
```

### Option 3: Build from Source (Advanced Users)

For advanced users who want optimal performance, building from source is recommended:

#### Install Dependencies in WSL2 Ubuntu
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt-get install build-essential git cmake wget libboost-all-dev \
    libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler \
    libhdf5-dev libatlas-base-dev python3-dev librdkit-dev python3-numpy \
    python3-pip python3-pytest libjsoncpp-dev

# Install CUDA (if you have NVIDIA GPU)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
chmod +x cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Install OpenBabel3
git clone https://github.com/openbabel/openbabel.git
cd openbabel
mkdir build && cd build
cmake -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON ..
make -j8
sudo make install
cd ../..

# Build GNINA
git clone https://github.com/gnina/gnina.git
cd gnina
mkdir build && cd build
cmake ..
make -j8
sudo make install
```

## Integration with PureProt

Once GNINA is installed, PureProt will automatically detect it:

```bash
# Test GNINA detection in PureProt
cd /path/to/PureProt
python -c "
from modeling.advanced_docking_engine import create_docking_engine
engine = create_docking_engine()
status = engine.get_engine_status()
print('GNINA available:', status['GNINA'])
print('Primary method:', status['primary_method'])
"
```

## Usage Examples

### Basic Docking with GNINA
```bash
# Dock a single ligand
gnina -r protein.pdb -l ligand.sdf -o output.sdf --autobox_ligand ligand.sdf

# Dock with custom binding site
gnina -r protein.pdb -l ligand.sdf -o output.sdf \
      --center_x 10.0 --center_y 15.0 --center_z 20.0 \
      --size_x 20 --size_y 20 --size_z 20
```

### Using GNINA with PureProt CLI
```bash
# Hybrid screening with GNINA (when available)
python PureProt.py hybrid-screen batch_molecules.csv \
    --protein protein.pdb \
    --center "10.0,15.0,20.0" \
    --size "20,20,20"
```

## Performance Notes

- **GNINA with GPU**: Significantly faster, requires CUDA-compatible GPU
- **GNINA CPU-only**: Slower but still more accurate than traditional methods
- **Fallback**: If GNINA is not available, PureProt uses RDKit shape matching

## Troubleshooting

### Common Issues

1. **"gnina: command not found"**
   - Ensure the binary is in your PATH
   - Try using the full path: `/usr/local/bin/gnina`

2. **CUDA errors**
   - Install appropriate CUDA version for your GPU
   - Use CPU-only version if no GPU available

3. **Memory issues**
   - Reduce batch size for large molecule sets
   - Use `--cnn_scoring` flag to control CNN usage

### Getting Help

- **GNINA GitHub**: https://github.com/gnina/gnina
- **GNINA Slack**: Subscribe to their team for support
- **Documentation**: https://gnina.github.io/gnina/

## Version Information

- **Latest Version**: v1.3.2 (as of 2024)
- **Minimum Requirements**: Ubuntu 22.04, CUDA 12.0+ (for GPU)
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM

## Integration Status in PureProt

✅ **Automatic Detection**: PureProt detects GNINA installation automatically  
✅ **Fallback Support**: Uses RDKit shape matching if GNINA unavailable  
✅ **Hybrid Scoring**: Combines GNINA with AI predictions for optimal results  
✅ **CLI Integration**: Full support in `dock` and `hybrid-screen` commands  

The advanced docking engine in PureProt provides a seamless experience whether GNINA is available or not, ensuring your research can proceed with the best available docking method.
