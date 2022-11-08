import os
from pathlib import Path
import sys
import argparse

import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser("Compare values with onnxruntime")
parser.add_argument("model_file")
parser.add_argument("--init", default="ones")
parser.add_argument("--output_file", default="output.txt")
args = parser.parse_args()

outputs = {}

def parse_name(name):
    name, shape = name.split('(')
    return name, [int(t) for t in shape[:-1].split(",")]

act_dir = Path("./activations")
for fname in os.listdir(act_dir):
    fpath = act_dir / fname
    name,_ = os.path.splitext(fpath.name)

    if name == "": break

    # print(name)
    outputs[name] = np.load(fpath)

import onnx
from pathlib import Path

model_path = Path(args.model_file).expanduser()
model = onnx.load(model_path)
# model.graph.output.extend(list(outputs.keys()))

# new_graph = onnx.helper.make_graph(model.graph.node, 'resnet', ['data'], list(outputs.keys()))

value_info_protos = []

inter_layers = list(outputs.keys())
names = []


shape_info = onnx.shape_inference.infer_shapes(model)
for idx, node in enumerate(shape_info.graph.value_info):
    if node.name in inter_layers:
        names.append(node.name)
        value_info_protos.append(node)

for output in model.graph.output:
    names.append(output.name)

model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
onnx.checker.check_model(model)
onnx.save(model, './test.onnx')

onnx_input = model.graph.input[0]
input_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in onnx_input.type.tensor_type.shape.dim)

debug_shapes = False 

#init = "arange"
init = args.init
#init = "from_file"
if init == "ones":
    input = np.ones(input_shape, dtype=np.float32)
elif init == "range":
    s = 1
    for i in input_shape:
        s *= i
    print(s)
    input = np.arange(s, dtype=np.float32).reshape(input_shape)
else:
    #input = np.load(Path("~/irisa/diffusers/latents.npy").expanduser())
    input = np.load(Path(init).expanduser())

sess = ort.InferenceSession("./test.onnx")
ort_outputs = sess.run(names, {onnx_input.name: input}) 

THRESHOLD = 1e-1

if debug_shapes:
    print("checking shapes")

    for i, k in enumerate(names):
        print(k, ort_outputs[i].shape)
    sys.exit(0)


print()
print("checking equality")

with open("./classes.txt") as f:
    classes = f.readlines()

# for (k, a), b in zip(outputs.items(), ort_outputs):
for i, k in enumerate(names):
    a = outputs[k]
    b = ort_outputs[i]

    not_equal = (np.abs(a - b) > THRESHOLD)
    if a.size != b.size:
        print("not equal", k, a.shape, a.size, b.shape, b.size)
        #break
    elif not_equal.sum() != 0:
        print(f"{not_equal.sum()}/{not_equal.size} not equal {k}")
        # for f in a.flatten()[0:10]:
        #     print(f"{f:1.03f} ", end="")
        # print()
        # for f in b.flatten()[0:10]:
        #     print(f"{f:1.03f} ", end="")
        # print()
        #break
    else:
        print(f"{k} ✅")
        topk = 2
        print(
            "ort",
            np.argsort(-b)[0,:topk],
            [classes[i] for i in np.argsort(-b)[0,:topk]],
        )
        print(
            "wnn",
            np.argsort(-a)[0,:5],
            [classes[i] for i in np.argsort(-a)[0,:topk]],
        )

