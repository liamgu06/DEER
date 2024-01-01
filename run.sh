

# train the DEER on Letter-high with q = 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --DS 'Letter-high' --partial_rate 0.1 --log_file 'results.txt' \
                      --epochs 100 --number_of_run 5 --use_bn 0

# train the DEER on COIL-DEL with q = 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --DS 'COIL-DEL' --partial_rate 0.1 --log_file 'results.txt' \
                      --epochs 200 --number_of_run 5 --use_bn 1

                      