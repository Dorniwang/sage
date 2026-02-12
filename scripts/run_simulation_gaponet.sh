python scripts/run_simulation_gaponet.py \
    --robot-name so101 \
    --motion-source custom \
    --motion-files /cpfs/workspace/code/sage_lerobot/sage/motion_files/so101/custom/custom_motion.txt \
    --output-folder ./gaponet_output_so101 \
    --gaponet-model /cpfs/workspace/data/pretrained_models/sim2real/gaponet/logs/rsl_rl/so101_operator/2026-02-09_17-38-24_run_013_nosim_with_random_v1_3/jit_models/model_51000.pt \
    --joint-delta-scale "1.0, 1.0, 1.0, 1.0, 0.2, 0.2" \
    # --headless
