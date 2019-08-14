import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

log_queue = None
process = None

def async_log(tensorboard_dir, log):
    global log_queue
    if log_queue is None or log is None:
        return
    finished = False
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    while True:
        if finished and log_queue.empty():
            tensorboard_writer.close()
            return

        temp = log_queue.get()
        if temp is None:
            return
        if temp == "finish":
            finished = True
            continue
        
        if temp[0] == "tensorboard":
            function, args = temp[1:]
            getattr(tensorboard_writer, function)(*args)
        elif temp[0] == "log":
            log.info(temp[1])

def initialize(tensorboard_dir, log):
    global log_queue
    log_queue = mp.Queue()
    global process
    process = mp.Process(target=async_log, args=[tensorboard_dir, log])
    process.start()

def write_tensorboard(func, args):
    global log_queue
    log_queue.put(("tensorboard", func, args))

def write_log(message):
    global log_queue
    log_queue.put(("log", message))

def close_thread(wait_to_finish=False):
    global process
    if process:
        try:
            if wait_to_finish:
                global log_queue
                if log_queue is not None:
                    log_queue.put("finish")
                process.join()
            else:
                process.close()
        except:
            process.terminate()

