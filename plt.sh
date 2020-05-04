cat run_kaggle_pt.log|grep "apply_emb time" > apply_emb
# run on gpu
cat run_kaggle_pt.log_gpu|grep "interact_features time" > interact_features
cat run_kaggle_pt.log_gpu|grep "apply_mlp time" > apply_mlp
gnuplot -p -e 'set logscale y;set xlabel "iteration";set ylabel "runtime(sec)";plot "apply_emb" u 0:3 w linesp title "apply emb on CPU", "interact_features" u 0:3 w linesp title "interact features on GPU", "apply_mlp" u 0:3 w linesp title "apply mlp on GPU"'
