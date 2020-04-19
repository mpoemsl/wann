# basic wann forestfires and mnist experiments

python train_wann.py forestfires --n_gen 200 --pop_size 100 --weight_type shared --prob_crossover 0.0 # 200 * 100 = 20_000 samples
python train_wann.py forestfires --n_gen 200 --pop_size 100 --weight_type random --prob_crossover 0.0 # 200 * 100 = 20_000 samples

python train_wann.py mnist --n_gen 30 --pop_size 50 --weight_type shared --prob_crossover 0.0 # 30 * 1000 = 30_000 samples
python train_wann.py mnist --n_gen 30 --pop_size 50 --weight_type random --prob_crossover 0.0 # 30 * 1000 = 30_000 samples


# basic ann forestfires and mnist experiments

python run_ann.py forestfires --n_epochs 49 # 410 * 49 = 20_090 samples (only 20_000 will be plotted)
python run_ann.py mnist --n_epochs 1 # 60_000 * 1 = 60_000 samples (only 30_000 will be plotted)
