rm -f apply_emb_cpu_delta apply_emb_gpu_delta E_cpu_delta E_gpu_delta
touch apply_emb_cpu_delta apply_emb_gpu_delta E_cpu_delta E_gpu_delta
for bs in 256 1024 4096 16384 65536
do
  echo -n ${bs} >>E_cpu_delta         ;grep ^E        run_kaggle_pt.log_bs${bs}_arch16     | awk '{ssy=ssy+$NF} END{print " " ssy}' >> E_cpu_delta
  echo -n ${bs} >>apply_emb_cpu_delta ;grep apply_emb run_kaggle_pt.log_bs${bs}_arch16     | awk '{ssy=ssy+$NF} END{print " " ssy}' >> apply_emb_cpu_delta
  echo -n ${bs} >>E_gpu_delta         ;grep ^E        run_kaggle_pt.log_bs${bs}_arch16_gpu | awk '{ssy=ssy+$NF} END{print " " ssy}' >> E_gpu_delta
  echo -n ${bs} >>apply_emb_gpu_delta ;grep apply_emb run_kaggle_pt.log_bs${bs}_arch16_gpu | awk '{ssy=ssy+$NF} END{print " " ssy}' >> apply_emb_gpu_delta
done

gnuplot -p -e 'set xlabel "batch size"; set ylabel "Run time for 200 iteration(sec)";plot "E_cpu_delta"  u 1:2 w linesp , "apply_emb_cpu_delta"  u 1:2 w linesp ,"E_gpu_delta"  u 1:2 w linesp  ,"apply_emb_gpu_delta"  u 1:2 w linesp '
