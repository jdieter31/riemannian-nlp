import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

tensorboard_queue = None
process = None

def async_tensorboard(tensorboard_dir):
    global tensorboard_queue
    if tensorboard_queue is None:
        return
    finished = False
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    while True:
        if finished and tensorboard_queue.empty():
            tensorboard_writer.close()
            return

        temp = tensorboard_queue.get()
        if temp is None:
            return
        if temp == "finish":
            finished = True
            continue

        function, args = temp
        getattr(tensorboard_writer, function)(*args)

def initialize(tensorboard_dir):
    global tensorboard_queue
    tensorboard_queue = mp.Queue()
    global process
    print(tensorboard_dir)
    process = mp.Process(target=async_tensorboard, args=[tensorboard_dir])
    process.start()

def write_tensorboard(func, args):
    global tensorboard_queue
    tensorboard_queue.put((func, args))

def close_thread(wait_to_finish=False):
    global process
    if process:
        try:
            if wait_to_finish:
                global tensorboard_queue
                if tensorboard_queue is not None:
                    tensorboard_queue.put("finish")
                process.join()
            else:
                process.close()
        except:
            process.terminate()

