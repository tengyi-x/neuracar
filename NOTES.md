# Notes

## Building libCacheSim (on linux.student.cs.uwaterloo.ca)

`glib`, `zstd`, `cmake`, `gcc`/`g++` were already present system-wide. Only `liblz4` and (for the LRB
baseline) LightGBM were missing:

```bash
# liblz4
cd ~/cs486/project
git clone --depth 1 https://github.com/lz4/lz4.git
cd lz4 && make -j$(nproc) && make install PREFIX=$HOME/cs486/project/local

# LightGBM (needed for the LRB baseline)
cd ~/cs486/project
git clone --recursive --depth 1 https://github.com/microsoft/LightGBM.git
mkdir -p LightGBM/build && cd LightGBM/build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/cs486/project/local -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && make install

# make both discoverable by pkg-config/cmake (add to ~/.bashrc)
export PKG_CONFIG_PATH=$HOME/cs486/project/local/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$HOME/cs486/project/local:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/cs486/project/local/lib:$LD_LIBRARY_PATH
```

Then build libCacheSim itself, with LRB enabled:

```bash
bash scripts/setup_libcachesim.sh
cd third_party/libCacheSim/_build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_LRB=ON -DCMAKE_PREFIX_PATH=$HOME/cs486/project/local
make -j$(nproc)
```

## Running `cachesim`

`--help`'s example (`obj-id-col=1;delimiter=,`) uses semicolons, but the actual parser
(`libCacheSim/bin/cli_reader_utils.c`) splits `-t`/`--trace-type-params` on **commas**, not
semicolons. Use commas:

```
-t "time-col=2,obj-id-col=5,size-col=4,delimiter=,,has-header=true"
```

### Bundled real traces (`data/`)

- `cloudPhysicsIO.csv` — columns: `version,time,op,size,lbn` (has header). Use
  `time-col=2,obj-id-col=5,size-col=4,delimiter=,,has-header=true`.
- `twitter_cluster52.csv` — columns: `time,object,size,next_access_vtime`. First line is a `#`-commented
  header row — use `has-header=true` to skip it (it's *not* header-less despite the `#`). Use
  `time-col=1,obj-id-col=2,size-col=3,delimiter=,,has-header=true`. On this trace with 100MiB/LRU: miss
  ratio 0.1435 over 1M requests.

### Confirmed baselines (on `cloudPhysicsIO.csv`, 100MiB cache)

All six of LRU, LFU, ARC, LeCaR, LHD, LRB run successfully:

| Algorithm | Miss ratio |
|---|---|
| LRU | 0.8210 |
| LFU | 0.8081 |
| ARC | 0.7967 |
| LeCaR | 0.8200 |
| LHD | 0.7902 |
| LRB | 0.8210 |

LRB needs `LD_LIBRARY_PATH` to include the LightGBM install dir at runtime, not just build time.
