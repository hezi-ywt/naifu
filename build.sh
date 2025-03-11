#  python data_loader/csv2arrow.py /home/ywt/lab/sd-scripts/library/vae_trainer/test dataset/porcelain/arrows 1
 # Single Resolution Data Preparation
idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json

 # Multi Resolution Data Preparation     
 idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json

#  使用参数指定
# idk multibase --src dataset/porcelain/jsons/porcelain.json --base-sizes 1024 --reso-step 64 --min-size 256 -t dataset/porcelain/jsons/porcelain_mt.json

# # 或使用配置文件
# idk multibase -c dataset/yamls/porcelain_mb.yaml -t dataset/porcelain/jsons/porcelain_mb.json