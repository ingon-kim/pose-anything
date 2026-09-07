# Install, run, and ship

> Moved verbatim from the top-level README on 2026-09-02, when the README was cut down to a front page. Nothing here was rewritten.

## Requirements

**With Docker** the host only needs an NVIDIA driver, Docker +
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and a Hugging Face account — everything below ships inside the image. For native installs:

- Ubuntu 24.04 (verified on WSL2) with **ROS 2 Jazzy**
- NVIDIA GPU with ≥ 6 GB VRAM, PyTorch CUDA build (bf16 inference). **That figure
  assumes a card with dedicated VRAM** — every memory number in this repository was
  measured on an RTX 4070 Ti (12,282 MiB) under WSL2. On unified-memory machines
  (e.g. DGX Spark / GB10) `nvidia-smi` reports no total at all, so this requirement
  does not translate as written: see [`docs/shared_server_2026-08-31.md`](shared_server_2026-08-31.md)
- Python 3.12 — `torch==2.10.0`, `transformers==5.5.0`, `open3d==0.19.0`, `rosbags`,
  `scipy`, `opencv-python` (the first three are pinned — see *Native install* below)
- Intel RealSense D455 + [`realsense2_camera`](https://github.com/IntelRealSense/realsense-ros) for live input
- A Hugging Face account with access to the gated
  [`facebook/sam3`](https://huggingface.co/facebook/sam3) checkpoint
  (click *"Agree and access repository"* on the model page — approval is immediate)


## Quick start

### Docker (recommended — any Ubuntu PC with an NVIDIA GPU)

```bash
git clone https://github.com/ingon1026/pose-anything.git
cd pose-anything
export HF_TOKEN=hf_xxxx            # token of an account with facebook/sam3 access
docker compose run --rm perception                                    # live camera
```

**A RealSense D455 is all you need.** `run.sh` with no arguments starts the camera
itself and runs the pipeline on it — that is the product path, and it needs no data
from this repository.

Rosbags are for **reproducing the measurements in `docs/` and running the regression
gate**, not for normal use. They are not distributed with this repo (see
[`docs/datasets.md`](datasets.md)); if you have one of your own, point at it:

```bash
docker compose run --rm perception ./run.sh path/to/your/bag --prompts "book"
```

The prebuilt image is pulled from
[Docker Hub (`ingon1026/pose-anything`)](https://hub.docker.com/r/ingon1026/pose-anything)
on first run — no local build needed. To build from source instead: `docker compose build` (~21 GB).

**On arm64 you have to build** — the Hub image is amd64-only, so a
plain `docker compose run` fails with *no matching manifest for linux/arm64*:

```bash
# .env — Compose reads it automatically; it is git-ignored.
#   IMAGE_TAG=1.3.0-arm64   ← the local build gets this name instead of :latest,
#                             which on Hub is amd64 and would collide.
#   UID / GID               ← see the comments in docker-compose.yml
./scripts/set_hf_token.sh hf_xxxx             # validates the token, then writes .env

docker compose build                          # BuildKit sets TARGETARCH=arm64 itself
docker compose run --build --rm perception
```

`IMAGE_TAG` is one line you live with: the architecture is something the machine already
knows, and carrying that `.env` to an x86 box breaks it. A multi-arch manifest would
remove it, but that needs the arm64 tag pushed to Hub, and on a single shared machine
that push buys nothing — the image is already in the local daemon. See
`docs/shared_server_2026-08-31.md` §9-5 for when pushing does start to pay.

`Dockerfile` swaps in the CUDA 13.0 PyTorch wheels and an Open3D wheel from upstream's
release page on that branch; nothing else changes. **None of the numbers in this README
were measured on arm64.**

**What the arm64 branch actually requires is CUDA 13.0**, not merely aarch64 — the wheels
are `+cu130` and the machine it was verified on runs driver 580.95.05. Nothing in the
`Dockerfile` is specific to GB10, so any aarch64 host with a driver new enough for CUDA 13.0
should work, and only DGX Spark has been tried. **Jetson is a different case and was
previously listed here in error**: JetPack pins the driver to its L4T BSP (CUDA 12.x at the
time of writing), so the `+cu130` wheels have no driver to run against, and PyTorch for
Tegra comes from NVIDIA's own index rather than `download.pytorch.org`. That reasoning is
from the version constraints, **not from a build attempt** — if you need Jetson, check
`nvidia-smi` for the CUDA version first; it will not be a matter of just building.

> **If you edited the source, you must pass `--build`.** The service declares both
> `build:` and `image:` with no `pull_policy`, so Compose *pulls first* and only builds
> when the pull fails ([Compose build spec](https://docs.docker.com/reference/compose-file/build/#using-build-and-image)).
> A plain `docker compose run` therefore runs the **Hub image, not your working tree** —
> silently, with nothing in the output to say so. Use `docker compose run --build --rm perception`.
> The published image can also simply be **older than this repository** — it is pushed by
> hand, not by CI. Current: **`1.3.0`** (= `latest`, digest `sha256:96657bbd72b6…`), built
> and pushed 2026-08-28 from commit `ada2aa8`. Verified inside the container: `174 passed`
> (same count as the host), `import yaml` resolves, CUDA visible, and `SAM3 ready` in 9.5 s
> — that last figure was taken with Isaac Sim rendering on the same GPU, so it is not
> comparable to the 4.1 s measured standalone on `1.2.0`. Only `src/`, `scripts/`,
> `run.sh` and the entrypoint go into the image, so documentation-only commits after
> that one do not make it stale.

> **If Isaac Sim is running, stop it or move to another ROS domain.** Its ROS 2 bridge
> publishes `/camera/camera/color/camera_info` — the same topic this node subscribes to —
> with a different resolution. The node sees the calibration flip back and forth, resets on
> every change (mixing calibrations would project pixels into the wrong 3D rays), and
> publishes nothing. Observed 2026-08-25: replaying a bag with Isaac open produced a
> header-only CSV and a log full of `camera_info changed`; `ROS_DOMAIN_ID=77` on both the
> node and the player fixed it in one run.
>
> **Domain isolation does not cover `run.sh`'s node cleanup.** `run.sh` runs
> `pkill -f "roboworld_perception/perception_node"` on startup, and again on exit.
> That match is at the process level and ignores `ROS_DOMAIN_ID`, so a `run.sh`
> invocation kills perception nodes on **every** domain, not just yours (measured).

Model weights are **not** baked into the image (Meta's gated license) — they are
downloaded once on first run into a mounted cache volume and reused afterwards.
Put rosbags under `./bags/` (mounted into the container); add `--headless` on
machines without a display.

### Native install

```bash
git clone https://github.com/ingon1026/pose-anything.git
cd pose-anything

# ROS 2 side. A ros-base install has none of these; ros-desktop still lacks
# vision_msgs and the mcap storage plugin. Without vision-msgs the node dies
# at import; without rosbag2-storage-mcap no bag in this project will replay
# (they are all recorded as mcap).
sudo apt install -y \
    ros-jazzy-vision-msgs \
    ros-jazzy-diagnostic-msgs \
    ros-jazzy-rosbag2-storage-mcap \
    ros-jazzy-rviz2 \
    ros-jazzy-realsense2-camera 'ros-jazzy-librealsense2*' \
    python3-colcon-common-extensions \
    fonts-noto-cjk \
    libgl1 libgomp1 usbutils

pip install --user --break-system-packages \
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install --user --break-system-packages \
    transformers==5.5.0 open3d==0.19.0 rosbags scipy opencv-python pillow
hf auth login                      # account with facebook/sam3 access
colcon build --symlink-install
```

The three pinned packages (`torch`, `transformers`, `open3d`) are the ones the
measured numbers above depend on; `torchvision` is pinned only because it has to
match `torch`. Moving any of them invalidates the measurement conditions —
re-run the regression gate above before trusting the table again.
`realsense2_camera` / `librealsense2` are needed only for live camera input;
`fonts-noto-cjk` only for Korean labels in the debug overlay.

> **The container pins more than this list, and you should not simply copy it.**
> `Dockerfile` also pins `numpy`, `scipy`, `opencv-python`, `pillow`, `rosbags`
> and `pytest`, because inside the image nothing else needs them. On a host where
> `apt` manages a ROS-side `numpy`, forcing those same versions shadows it — the
> exact "value measured under one condition applied to another" trap this repo has
> been bitten by. Pin them on the host only if you are reproducing a measurement
> and know what you are shadowing.
>
> Two of those pins were found the hard way on 2026-08-26 and apply to a native
> install too: `opencv-python` uses a four-part version (`4.13.0.92`; `cv2.__version__`
> reports the three-part library version and is *not* a valid pin), and `pytest`
> **9.1.1 cannot start at all in a ROS-sourced shell** — Jazzy's `launch_testing`
> plugin uses an older hook signature. Use `pytest==9.0.2` if you intend to run the
> test suite below.

```bash
./run.sh                           # live RealSense camera
./run.sh path/to/rosbag            # play a recorded bag
./run.sh path/to/rosbag --prompts "book,glove"   # non-interactive
```

The script asks one question — which objects to find. Type a name like `book`
(or a Korean alias like `책`) and **only the prompted objects are detected and
tracked** — one best-scoring instance per prompt, with a persistent track ID.
Everything else in the scene is ignored.

It then starts the perception node, RViz (preset with 3D boxes and axes), and a
full-size debug window. Per-frame results are logged to `output/ros_<timestamp>.csv`.

Any run that opens RViz also turns `publish_world_tf` on — the preset's Fixed
Frame is `world`, and that frame exists only because of this TF. Without it RViz
draws nothing and reports no error. (`--headless` leaves it off, and so does any
launch with `rviz:=false`.)

Offline processing (no ROS runtime — reads the bag directly, writes mp4 + CSV):

```bash
python3 scripts/run_offline.py --bag path/to/rosbag --prompts "book" --show
```


### Prompts

SAM 3's text encoder is English-based. These Korean words are translated
automatically (see `PROMPT_ALIASES` in `sam3_detector.py`); use English for
anything else:

> 물통 · 마우스 · 필통 · 노트북 · 책 · 스마트폰 · 장갑 · 천 · 블록

**If detection is unstable, change the word before touching the threshold.**
The same green thermos scores 0.45 as `"water bottle"` but 0.91 as `"thermos"`.
Prompts can be swapped at runtime:

```bash
ros2 topic pub --once /perception/prompt std_msgs/String "data: thermos"
```

**Don't know good prompt words for a new bag?** Suggest them from a few frames
with a local VLM + SAM3 scoring (one-time setup, not for runtime):
`python3 scripts/suggest_prompts.py --bag path/to/rosbag` — prints a
comma-separated prompt string on its last line, ready for `--prompts` above.
See `docs/vlm_prompt_spike_2026-09-04.md` for validation (21/22 at/above manual score).


## Regression gate before shipping

There is no automated accuracy gate in CI — the bags and the reference CSVs are
not in this repository (`bags/`, `output/` are gitignored), so this has to be run
by hand on a machine that has them:

1. **Reference** — replay a bag on a known-good build and keep the CSV. **Pass the
   shipping preset explicitly** — the script's own defaults are *not* what ships
   (`image_size` 1008 vs 672, `detect_interval` 5 vs 1, and `max_per_prompt` **1**,
   which silently gates the 8-block Isaac scene down to a single track and exercises
   neither merging nor the footprint gate). The values below mirror `isaac.launch.py`:

   ```bash
   python3 scripts/run_offline.py --bag bags/isaac_belt_moving \
       --prompts "blue bar with holes" \
       --image-size 672 --detect-interval 1 --max-per-prompt 10 \
       --out output/ref_<date>
   ```
2. **Candidate** — replay the *same* bag on the build you intend to ship, into a
   different `--out` directory.
3. **Judge** — `python3 scripts/check_accuracy.py --ref output/ref_<date> --cand output/<candidate>`
   must print **PASS**. Anything else is a regression, not a rounding difference —
   the thresholds are derived from this pipeline's own frame-to-frame jitter
   (see the constants block at the top of that script). Use the Isaac bag: the
   script's absolute floor is derived for that scene, and the larger run-to-run
   plane spread of test2/test4 produces spurious FAILs at it.

Regenerate the reference after any change to `geometry.py` / plane fitting or to
a pinned dependency version — an old reference silently blesses the old behaviour.


## Project structure

```
src/roboworld_perception/         ROS 2 package (ament_python)
  roboworld_perception/
    sam3_detector.py              SAM 3 wrapper, prompt aliases
    pipeline.py                   hybrid detect/track orchestration
    tracker.py                    IoU association, track lifecycle, rotation stabilization
    fusion.py                     per-track Kalman filter + χ² observation gating
    geometry.py                   back-projection, OBB, axis matching
    overlay.py                    debug rendering
    input_health.py               input contract checks, 1 Hz diagnostics
    pose_covariance.py            position covariance for PoseWithCovariance
    perception_node.py            ROS 2 node
  launch/                         perception.launch.py (base) · isaac.launch.py (Isaac preset)
  rviz/perception.rviz            RViz preset (Fixed Frame `world`)
  test/                           pytest, 16 files incl. filter property tests
                                  (`pytest src/roboworld_perception/test -q`;
                                   `scripts/check_ci_env.sh` reproduces CI)
scripts/run_offline.py            bag → mp4/CSV without ROS
run.sh                            single entry point
```

