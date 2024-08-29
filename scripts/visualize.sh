source activate
conda activate HUMANISE

cd ..
rm -r ./visualize.log
# nohup python visualize.py > visualize.log \
PYOPENGL_PLATFORM=egl \
python visualize.py \
--input_path ./datasets/demo_scene/ScanNet0604 \
--fps 20 \
--scene_name Demo \
--output_obj False
