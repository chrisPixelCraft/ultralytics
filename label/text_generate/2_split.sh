# cams=(Cam1 Cam2 Cam3 Cam4 Cam5 Cam6 Cam7 Cam8 Cam9 Cam10 Cam11)
# cams=(Cam1 Cam3 Cam4 Cam10 Cam11)
cams=(Cam10)

for cam in ${cams[@]}
do
    python3 gather.py \
        --input_pth /home/chenyukai/NTU-dataset/label/text_generation/afterByte \
        --output_pth ./split \
        --camid ${cam}
done