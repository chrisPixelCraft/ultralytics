# cams=(Cam1 Cam2 Cam3 Cam4 Cam5 Cam6 Cam7 Cam8 Cam9 Cam10 Cam11)
# cams=(Cam1 Cam3 Cam4 Cam10 Cam11)
# cams=(Cam2 Cam3 Cam6 Cam8 Cam9 Cam10 Cam11)
cams=(Cam4 Cam10 Cam11)

for cam in ${cams[@]}
do
    python3 roi.py \
        --input_dir ../ByteTrack/ \
        --input_msk ../mask_generation/ROI/ \
        --output_dir ./afterByte \
        --camid ${cam}
    
    python3 gather.py \
        --input_pth /home/chenyukai/NTU-dataset/label/text_generation/afterByte \
        --output_pth ./split \
        --camid ${cam}
done