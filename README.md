# Fast-GraspD – Differentiable Grasp Optimisation
[Project](https://dexgrasp.github.io/) | [Paper](https://ieeexplore.ieee.org/document/10160314)

Fast-GraspD searches for stable grasps on a given object mesh using a
fully-differentiable Warp simulation.  **Only the Allegro right hand is
supported at the moment.**

---

## Install

```bash
pip install -r requirements.txt  # tested with warp-lang==1.7.1, python 3.12
```

---

## Run

```bash
python scripts/collect_grasps_allegro.py \
  collector_config.obj_set=ycb \
  collector_config.obj_name=003_cracker_box
```

All parameters live in `conf/collect_grasps/config.yaml` and can be overridden
from the command line via Hydra (`collector_config.*`).  Tune the loss weights
(`w_*_loss`) for best results.

---

## Outputs

```
<output_dir>/<hand_name>_<obj_set>/<obj_name>/
    *.npy              # final joint angles
    *.usd              # final grasp (optional)
    *_opt_traj.usd     # optimisation trajectory (optional)
```

USD export is toggled by `render_final_grasp` and `render_opt_traj` in the
config. 

---

## TODO

- [ ] Add support for Barrett hand
- [ ] Add support for Shadow hand
- [ ] Set tested default configuration values
