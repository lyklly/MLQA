# label sharing MiniImagenet 1shot
python main.py --share=1 --update_batch_size=1 --update_batch_size_eval=1 --metatrain_iterations=50000 > shmini-mlqa-1.log
# label sharing MiniImagenet 5shot
python main.py --share=1 --update_batch_size=5 --update_batch_size_eval=5 --metatrain_iterations=50000 > shmini-mlqa-5.log


# non label sharing MiniImagenet 1shot
python main.py --share=0 --update_batch_size=1 --update_batch_size_eval=15 --metatrain_iterations=50000 > mini-mlqa-1.log
# non label sharing MiniImagenet 1shot
python main.py --share=0 --update_batch_size=5 --update_batch_size_eval=15 --metatrain_iterations=50000 > mini-mlqa-5.log
