rm -f apply_emb_bs        
rm -f interact_features_bs
rm -f apply_mlp_bs        
rm -f zero_grad_bs
rm -f backward_bs
rm -f optimizer_bs
rm -f apply_emb_bs_gpu        
rm -f interact_features_bs_gpu
rm -f apply_mlp_bs_gpu        
rm -f zero_grad_bs_gpu
rm -f backward_bs_gpu
rm -f optimizer_bs_gpu

touch apply_emb_bs        
touch interact_features_bs
touch apply_mlp_bs        
touch zero_grad_bs
touch backward_bs
touch optimizer_bs
touch apply_emb_bs_gpu        
touch interact_features_bs_gpu
touch apply_mlp_bs_gpu        
touch zero_grad_bs_gpu
touch backward_bs_gpu
touch optimizer_bs_gpu

for bs in 256 1024 4096 16384 65536
do 
  echo -n ${bs} >> apply_emb_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "apply_emb time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> apply_emb_bs        
  echo -n ${bs} >> interact_features_bs     ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "interact_features time"  | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> interact_features_bs
  echo -n ${bs} >> apply_mlp_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "apply_mlp time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> apply_mlp_bs        
  echo -n ${bs} >> zero_grad_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "zero_grad time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> zero_grad_bs        
  echo -n ${bs} >> backward_bs              ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "backward time"           | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> backward_bs        
  echo -n ${bs} >> optimizer_bs             ;cat run_kaggle_pt.log_bs${bs}_arch16     |grep "optimizer time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> optimizer_bs

  echo -n ${bs} >> apply_emb_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "apply_emb time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> apply_emb_bs_gpu        
  echo -n ${bs} >> interact_features_bs_gpu ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "interact_features time"  | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> interact_features_bs_gpu
  echo -n ${bs} >> apply_mlp_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "apply_mlp time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> apply_mlp_bs_gpu
  echo -n ${bs} >> zero_grad_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "zero_grad time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> zero_grad_bs_gpu        
  echo -n ${bs} >> backward_bs_gpu          ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "backward time"           | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> backward_bs_gpu        
  echo -n ${bs} >> optimizer_bs_gpu         ;cat run_kaggle_pt.log_bs${bs}_arch16_gpu |grep "optimizer time"          | awk '{ssy=ssy+$NF;cnt=cnt+1} END{print " " ssy/cnt}'>> optimizer_bs_gpu
done


gnuplot -p -e 'set title "emb vec len 16 on CPU";set logscale xy;set yrange[0.0001:1];set xlabel "batch size";set ylabel "run time(sec)";plot 
  "apply_emb_bs"              u 1:2 w linesp, 
  "interact_features_bs"      u 1:2 w linesp, 
  "apply_mlp_bs"              u 1:2 w linesp, 
  "zero_grad_bs"              u 1:2 w linesp, 
  "backward_bs"      u 1:2 w linesp,
  "optimizer_bs"              u 1:2 w linesp
'


gnuplot -p -e 'set title "emb vec len 16 on GPU";set logscale xy;set yrange[0.0001:1];set xlabel "batch size";set ylabel "run time(sec)";plot 
  "apply_emb_bs_gpu"          u 1:2 w linesp, 
  "interact_features_bs_gpu"  u 1:2 w linesp, 
  "apply_mlp_bs_gpu"          u 1:2 w linesp,
  "zero_grad_bs_gpu"          u 1:2 w linesp, 
  "backward_bs_gpu"  u 1:2 w linesp, 
  "optimizer_bs_gpu"          u 1:2 w linesp 
'
