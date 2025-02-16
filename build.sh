#  python data_loader/csv2arrow.py /home/ywt/lab/sd-scripts/library/vae_trainer/test dataset/porcelain/arrows 1
 # Single Resolution Data Preparation
 idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json

 # Multi Resolution Data Preparation     
 idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json