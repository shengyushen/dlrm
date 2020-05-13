cat $1 |grep "apply_emb time" > apply_emb
cat $1 |grep "interact_features time" > interact_features
cat $1 |grep "apply_mlp time" > apply_mlp
# run on gpu
cat ${1}_gpu|grep "apply_emb time" > apply_emb_gpu
cat ${1}_gpu|grep "interact_features time" > interact_features_gpu
cat ${1}_gpu|grep "apply_mlp time" > apply_mlp_gpu
gnuplot -p -e 'set yrange [0.0001:1];set title "CPU run time";set logscale y;set xlabel "iteration";set ylabel "runtime(sec)";plot "apply_emb"     u 0:3 w linesp title "apply emb on CPU", "interact_features"     u 0:3 w linesp title "interact features on CPU", "apply_mlp"     u 0:3 w linesp title "apply mlp on CPU"'
gnuplot -p -e 'set yrange [0.0001:1];set title "GPU run time";set logscale y;set xlabel "iteration";set ylabel "runtime(sec)";plot "apply_emb_gpu" u 0:3 w linesp title "apply emb on GPU", "interact_features_gpu" u 0:3 w linesp title "interact features on GPU", "apply_mlp_gpu" u 0:3 w linesp title "apply mlp on GPU"'
gnuplot -p -e 'set yrange [0.0001:1];set title "CPU emb and GPU feature mlp run time";set logscale y;set xlabel "iteration";set ylabel "runtime(sec)";plot "apply_emb" u 0:3 w linesp title "apply emb on CPU", "interact_features_gpu" u 0:3 w linesp title "interact features on GPU", "apply_mlp_gpu" u 0:3 w linesp title "apply mlp on GPU"'
