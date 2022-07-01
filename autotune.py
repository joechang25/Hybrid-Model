import os
import argparse
import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime

import models
from models.utils import *
from utils.resize_utils import *
from utils.resize_dataset import *

import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)

model_names = sorted(models.model_class.keys())
supported_optimizer = ['sgd', 'sgdw', 'adam', 'adamw']
supported_print_format = ['tf', 'torch']

parser = argparse.ArgumentParser(description='Tensorflow ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-s', '--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0=auto)')
parser.add_argument('-t', '--test', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--optim', default='sgdw', type=str, metavar='OPT',
                    help='optimizer: ' +
                        ' | '.join(supported_optimizer) +
                        ' (default: sgdw)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='WD', help='initial weight decay', dest='wd')
parser.add_argument('--dropout', default=0.0, type=float, metavar='D',
                    help='dropout rate')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='I', help='image size (default: 224)')
parser.add_argument('--print-format', default='tf', type=str, metavar='F',
                    help="print format: tf | torch (default: tf)")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the checkpoint (default: none)')
parser.add_argument('--save-dir', default='saved_model', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ./saved_model)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--use-xla', dest='use_xla', action='store_true',
                    help='use xla compilation')
parser.add_argument('--mixed', dest='mixed', action='store_true',
                    help='use mixed precision')
parser.add_argument('--torch-init', dest='torch_init', action='store_true',
                    help='use torch layer initialization')
parser.add_argument('--summary', dest='summary', action='store_true',
                    help='print model summary')
parser.add_argument('--grad-accum', default=1, type=int, metavar='N',
                    help='number of batches for gradient accumulation (default: 1)')
parser.add_argument('--resize',action='store_true')

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def parse_options():
    args = parser.parse_args()

    if args.optim not in supported_optimizer:
        raise ValueError("Invalid optimizer '{}'".format(args.optim))
    if args.print_format not in supported_print_format:
        raise ValueError("Invalid print format '{}'".format(args.print_format))
    if args.resume and not os.path.exists(args.resume):
        raise ValueError('Resume but model file does not exist')
    if args.workers < 0:
        raise ValueError("Invalid number of workers '{}'".format(args.workers))
    if args.grad_accum < 1:
        raise ValueError("Invalid gradient accumulation number '{}'".format(args.grad_accum))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print('  => using CPU, this will be slow')

    if args.mixed:
        print('  => using mixed precision')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if args.grad_accum > 1:
        args.batch_size *= args.grad_accum
        print("  => using gradient accumulation; resetting "
              "batch size to '{}'".format(args.batch_size))

    if args.use_xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)

    if args.print_format == 'tf':
        args.print_freq = 1

    return args


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
#target = tvm.target.Target("llvm -mcpu=skylake-avx512", host="llvm")
#dev = tvm.device(str(target), 0)

target = "llvm"
dev = tvm.cpu(0)

batch_size = 1
dtype = "float32"
network = "efficientnetb4_95"
log_file = "%s.log" % network
graph_opt_sch_file = "%s_graph_opt.log" % network

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "input_1"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 16
os.environ["TVM_NUM_THREADS"] = str(num_threads)



tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=15, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)



def evaluate_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    
    #create model
    args = parse_options()
    args.arch = "efficientnet_b4"
    args.weights = "imagenet"
    args.pretrained = True
    args.image_size = 95
    resize_model = get_model(1000, args)
    #custom_ob = {'hard_swish' : models.mobilenet_v3.hard_swish}
    #resize_model = tf.keras.models.load_model("/home/joechang/resnet50-ckpt-39", custom_objects=custom_ob, compile=False)
    resize_model.summary()
    shape_dict = {"input_1": (1, 3, 95, 95)} 
    mod, params = relay.frontend.from_keras(resize_model, shape_dict)

    #mod, params, input_shape, out_shape = get_network(network, batch_size=1)


    input_shape = (1, 3, 95, 95) 
    output_shape = (1, 1000)

    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning kernel...")
    tune_kernels(tasks, **tuning_opt)
    print("Tuning graph...")
    tune_graph(mod["main"], input_shape, log_file, graph_opt_sch_file)

    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, input_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, input_shape)

    # compile kernels with graph-level best records
    print("\nEvaluation of the network been tuned on graph level:")
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        evaluate_performance(lib, input_shape)	

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
tune_and_evaluate(tuning_option)
