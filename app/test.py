import psutil
import os
process = psutil.Process()
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
