
# basic forestfires and mnist experiments

python train.py forestfires --n_gen 256 --pop_size 128 --weight_type shared --prob_crossover 0.0
python train.py forestfires --n_gen 256 --pop_size 128 --weight_type random --prob_crossover 0.0

python train.py mnist --n_gen 128 --pop_size 64 --weight_type shared --prob_crossover 0.0
python train.py mnist --n_gen 128 --pop_size 64 --weight_type random --prob_crossover 0.0

# basic forestfires and mnist experiments with crossover

python train.py forestfires --n_gen 256 --pop_size 128 --weight_type shared --prob_crossover 0.5
python train.py forestfires --n_gen 256 --pop_size 128 --weight_type random --prob_crossover 0.5

python train.py mnist --n_gen 128 --pop_size 64 --weight_type shared --prob_crossover 0.5
python train.py mnist --n_gen 128 --pop_size 64 --weight_type random --prob_crossover 0.5

# basic forestfires and mnist experiments at double n_gen

python train.py forestfires --n_gen 512 --pop_size 128 --weight_type shared --prob_crossover 0.0
python train.py forestfires --n_gen 512 --pop_size 128 --weight_type random --prob_crossover 0.0

python train.py mnist --n_gen 256 --pop_size 64 --weight_type shared --prob_crossover 0.0
python train.py mnist --n_gen 256 --pop_size 64 --weight_type random --prob_crossover 0.0

# basic forestfires and mnist experiments at double n_gen with crossover

python train.py forestfires --n_gen 512 --pop_size 128 --weight_type shared --prob_crossover 0.5
python train.py forestfires --n_gen 512 --pop_size 128 --weight_type random --prob_crossover 0.5

python train.py mnist --n_gen 256 --pop_size 64 --weight_type shared --prob_crossover 0.5
python train.py mnist --n_gen 256 --pop_size 64 --weight_type random --prob_crossover 0.5

