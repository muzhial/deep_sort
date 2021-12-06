# para=("001" "002" "003" "006" "009" "010", "011", "012", "013")
para=("010", "011")
for i in ${para[*]}; do
python car_tracking_server.py \
    --VIDEO_PATH  /dataset/mz/code/yoloxServer-vehicle/record1/$i.mp4 \
    --counting_line
done
