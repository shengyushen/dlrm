rm -f apply_emb_bs        
rm -f interact_features_bs
rm -f apply_top_bs        
rm -f apply_bot_bs        
rm -f zero_grad_bs
rm -f backward_bs
rm -f optimizer_bs
rm -f apply_emb_bs_gpu        
rm -f interact_features_bs_gpu
rm -f apply_top_bs_gpu        
rm -f apply_bot_bs_gpu        
rm -f zero_grad_bs_gpu
rm -f backward_bs_gpu
rm -f optimizer_bs_gpu

touch apply_emb_bs        
touch interact_features_bs
touch apply_top_bs        
touch apply_bot_bs        
touch zero_grad_bs
touch backward_bs
touch optimizer_bs
touch apply_emb_bs_gpu        
touch interact_features_bs_gpu
touch apply_top_bs_gpu        
touch apply_bot_bs_gpu        
touch zero_grad_bs_gpu
touch backward_bs_gpu
touch optimizer_bs_gpu

for bs in 256 1024 4096 16384 65536
do 
  echo -n ${bs} >> apply_emb_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "apply_emb time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> apply_emb_bs        
  echo -n ${bs} >> interact_features_bs     ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "interact_features time"  |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> interact_features_bs
  echo -n ${bs} >> apply_top_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "apply_top time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> apply_top_bs        
  echo -n ${bs} >> apply_bot_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "apply_bot time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> apply_bot_bs        
  echo -n ${bs} >> zero_grad_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "zero_grad time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> zero_grad_bs        
  echo -n ${bs} >> backward_bs              ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "backward time"           |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> backward_bs        
  echo -n ${bs} >> optimizer_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "optimizer time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> optimizer_bs

  echo -n ${bs} >> apply_emb_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "apply_emb time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> apply_emb_bs_gpu        
  echo -n ${bs} >> interact_features_bs_gpu ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "interact_features time"  |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> interact_features_bs_gpu
  echo -n ${bs} >> apply_top_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "apply_top time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> apply_top_bs_gpu
  echo -n ${bs} >> apply_bot_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "apply_bot time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> apply_bot_bs_gpu
  echo -n ${bs} >> zero_grad_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "zero_grad time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> zero_grad_bs_gpu        
  echo -n ${bs} >> backward_bs_gpu          ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "backward time"           |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> backward_bs_gpu        
  echo -n ${bs} >> optimizer_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "optimizer time"          |awk '{print $NF}' | grep -v "e"| sort -n -r | awk '{if(cnt>50) {ssy=ssy+$NF};cnt=cnt+1} END{print " " ssy/(cnt-50)}'>> optimizer_bs_gpu
done


gnuplot -p -e 'set title "emb vec len 16 on CPU";set logscale xy;set yrange[0.0001:1];set xlabel "batch size";set ylabel "run time(sec)";plot 
  "apply_emb_bs"              u 1:2 w linesp, 
  "interact_features_bs"      u 1:2 w linesp, 
  "apply_top_bs"              u 1:2 w linesp, 
  "apply_bot_bs"              u 1:2 w linesp, 
  "zero_grad_bs"              u 1:2 w linesp, 
  "backward_bs"      u 1:2 w linesp,
  "optimizer_bs"              u 1:2 w linesp
'


gnuplot -p -e 'set title "emb vec len 16 on GPU";set logscale xy;set yrange[0.0001:1];set xlabel "batch size";set ylabel "run time(sec)";plot 
  "apply_emb_bs_gpu"          u 1:2 w linesp, 
  "interact_features_bs_gpu"  u 1:2 w linesp, 
  "apply_top_bs_gpu"          u 1:2 w linesp,
  "apply_bot_bs_gpu"          u 1:2 w linesp,
  "zero_grad_bs_gpu"          u 1:2 w linesp, 
  "backward_bs_gpu"  u 1:2 w linesp, 
  "optimizer_bs_gpu"          u 1:2 w linesp 
'
