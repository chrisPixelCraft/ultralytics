# cover the mask on the frames
cams=(Cam6 Cam8)

for cam in ${cams[@]}
do
    python3 cover_msk.py \
    --input_dir /home/chenyukai/NTU-dataset/video/short_video/${cam}/imgs_15 \
    --input_msk ./ROI/${cam}_15_roi.jpg \
    --output_dir /home/chenyukai/NTU-dataset/video/short_video/${cam}/imgs_15_roi
done

# python3 cover_msk.py \
#     --input_dir /home/chenyukai/NTU-dataset/video/short_video/Cam10/imgs_15 \
#     --input_msk ./ROI/Cam10_15_roi.jpg \
#     --output_dir /home/chenyukai/NTU-dataset/video/short_video/Cam10/imgs_15_roi