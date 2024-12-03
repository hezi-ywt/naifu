
conda activate sdxlroot

pkill -f trainer


export http_proxy=http://214.28.51.55:13128
export https_proxy=http://214.28.51.55:13128
export HF_ENDPOINT="https://hf-mirror.com"

pip install waifuset

cp -rn /app/hfd/ws_arrow2 /data/sdxl/
cp -rn /app/hfd/ws_arrow3 /data/sdxl/
cp -rn /app/hfd/artist_arrow_dir /data/sdxl/
cp -r /app/naifu /data/

cd /data/naifu/
bash build.sh