# Repositories to Download

Download and install repositories given below:

1. [SAM 2](https://github.com/facebookresearch/sam2)

```bash
git clone https://github.com/facebookresearch/sam2
cd sam2
pip install -e ".[notebooks]"
```

2. [Open3D](https://github.com/isl-org/Open3D)

```bash
git clone https://github.com/isl-org/Open3D
cd Open3D

# Only needed for Ubuntu
util/install_deps_ubuntu.sh

mkdir build
cd build
cmake ..

# On Ubuntu
make -j$(nproc)

# On macOS
make -j$(sysctl -n hw.physicalcpu)

# Activate the virtualenv first
# Install pip package in the current python environment
make install-pip-package

# Verify installation
python -c "import open3d"
```
