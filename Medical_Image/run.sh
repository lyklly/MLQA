# isic 1shot
python main.py --datasource=isic --update_batch_size=1 --update_batch_size_eval=15 --metatrain_iterations=50000 > isic-mlqa-1.log
# isic 5shot
python main.py --datasource=isic --update_batch_size=5 --update_batch_size_eval=15 --metatrain_iterations=50000 > isic-mlqa-5.log

# dermnet 1shot
python main.py --datasource=dermnet --update_batch_size=1 --update_batch_size_eval=15 --metatrain_iterations=50000 > derm-mlqa-1.log
# dermnet 1shot
python main.py --datasource=dermnet --update_batch_size=5 --update_batch_size_eval=15 --metatrain_iterations=50000 > derm-mlqa-5.log
