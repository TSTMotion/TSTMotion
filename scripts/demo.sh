source activate
conda activate tstmotion

cd ..
python demo.py \
--prompt_path ./datasets/prompt \
--model_path ./OmniControl/save/omnicontrol_ckpt/model_humanml3d.pt \
--GPT_version gpt-4o \
--api_key  \
--scene_name ./datasets/demo_scene/ScanNet0604 \
--scene_pc ./datasets/demo_scene/ScanNet0604/scene0604_00_vh_clean.ply \
--scene_seg ./datasets/demo_scene/ScanNet0604/detection_results.pkl \
--num_samples 1 \
--caption "walk to the door" \
