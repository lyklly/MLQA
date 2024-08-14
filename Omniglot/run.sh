# Omniglot 1shot
python main.py --datasource=omniglot --update_lr=0.1 --meta_lr=0.005 --update_batch_size=1 --update_batch_size_eval=1 --metatrain_iterations=50000 > omi-mlqa-1.log
# Omniglot 5shot
python main.py --datasource=omniglot --update_lr=0.1 --meta_lr=0.005 --update_batch_size=5 --update_batch_size_eval=5 --metatrain_iterations=50000 > omi-mlqa-5.log

