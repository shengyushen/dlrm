#!/bin/bash

log_file=$1

time_calc()
{
    local match_line=$1
    local cnt=0
    local time_cnt=0

    grep "$match_line" $log_file > tmp.txt

    while read -r line 
    do
        local time_cnt_tmp=`echo $line | awk -F '=' '{print $2}'`
        time_cnt=$(printf "%.9f" `echo "scale=9;$time_cnt_tmp+$time_cnt" | bc`)
        #echo "$cnt:$time_cnt"
        cnt=$(($cnt+1))
    done < tmp.txt
    
    result=$(printf "%0.9f" `echo "scale=9;($time_cnt*1000000)/$cnt" | bc`)
    echo "$match_line=$result us"
}

sed -i "s/time /time=/g" $log_file

time_calc "apply_bot time"
time_calc "apply_emb time"
time_calc "interact_features time"
time_calc "apply_top time"
