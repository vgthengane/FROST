# FINAL_METHOD_BUFFER.md — **PRidE** buffer twin (DONE)

Canonical: **`_notes/pride_final.md`**.

Same as accepted **PRidE** buffer-free, only change: exemplar **buffer** on  
(`model_name=selfevo`, `fscil_preset=canonical`).

Still: close–far on **all 6**, max, base-only **ep10**, ridge_cache **α=0.5** (flat), static, train-CM conflict replay, seed 2008.

### In-domain
`95.3,93.7,93.8,93.2,91.4,3.9,92.1,91.0,89.9,89.7,89.4,88.6,89.0,3.0,80.4,74.4,72.5,71.0,69.5,69.4,11.0`

### Cross (close–far)
`92.2,91.1,90.5,88.9,88.5,87.5,86.4,85.3,84.6,83.2,81.9,10.3,92.7,91.2,88.4,87.2,5.5,93.6,92.6,91.6,91.1,2.5`

```bash
sbatch scripts/simpl_runs_pride_alpha05_buf_ep10.sh
```

Logs: `_experiments/00_ABLATION_ALPHA05_BUF_EP10/final10_*_hybrid3_st_buf_a05.log`
