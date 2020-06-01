cat $1 |grep "apply_emb time" > apply_emb
cat $1 |grep "interact_features time" > interact_features
cat $1 |grep "apply_top time" > apply_top
cat $1 |grep "apply_bot time" > apply_bot
cat $1 |grep "zero_grad time" > zero_grad
cat $1 |grep "backward time"  > backward
cat $1 |grep "optimizer time" > optimizer
# run on gpu
cat ${1}_gpu|grep "apply_emb time" > apply_emb_gpu
cat ${1}_gpu|grep "interact_features time" > interact_features_gpu
cat ${1}_gpu|grep "apply_top time" > apply_top_gpu
cat ${1}_gpu|grep "apply_bot time" > apply_bot_gpu
cat ${1}_gpu|grep "zero_grad time" > zero_grad_gpu
cat ${1}_gpu|grep "backward time"  > backward_gpu
cat ${1}_gpu|grep "optimizer time" > optimizer_gpu

gnuplot -p -e '
   set terminal png size 2048,1024;
   set output "tmp.png";
   set size 1,1;
   set multiplot;
   set size 0.5, 1;
   set origin 0.0,0.0;
   set yrange [0.0001:1];
   set title "CPU run time";
   set logscale y;
   set xlabel "iteration";
   set ylabel "runtime(sec)";
   plot "apply_emb"     u 0:3 w linesp title "apply emb on CPU", "interact_features"     u 0:3 w linesp title "interact features on CPU", "apply_top"     u 0:3 w linesp title "apply top on CPU" , "apply_bot"     u 0:3 w linesp title "apply bot on CPU", "zero_grad" u 0:3 w linesp title "zero_grad on CPU" , "backward" u 0:3 w linesp title "backward on CPU", "optimizer" u 0:3 w linesp title "optimizer on CPU";

   set origin 0.5,0.0;
   set yrange [0.0001:1];
   set title "GPU run time";
   set logscale y;
   set xlabel "iteration";
   unset ylabel;
   plot "apply_emb_gpu" u 0:3 w linesp title "apply emb on GPU", "interact_features_gpu" u 0:3 w linesp title "interact features on GPU",  "apply_top_gpu" u 0:3 w linesp title "apply top on GPU", "apply_bot_gpu" u 0:3 w linesp title "apply bot on GPU", "zero_grad_gpu" u 0:3 w linesp title "zero_grad on GPU" , "backward_gpu" u 0:3 w linesp title "backward on GPU", "optimizer_gpu" u 0:3 w linesp title "optimizer on GPU";
'
