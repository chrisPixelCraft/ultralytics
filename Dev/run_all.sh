export CUDA_VISIBLE_DEVICES=0
python3 pipeline.py -D -R -T -J ./setting_Gateway1.json  -RW reid_weight/v17.pth --configfile reid_weight/v17.yaml
python3 pipeline.py -D -R -T -J ./setting_Gateway2.json  -RW reid_weight/v17.pth --configfile reid_weight/v17.yaml
python3 pipeline.py -D -R -T -J ./setting_Gateway3.json  -RW reid_weight/v17.pth --configfile reid_weight/v17.yaml
python3 pipeline.py -D -R -T -J ./setting_Gateway4.json  -RW reid_weight/v17.pth --configfile reid_weight/v17.yaml
