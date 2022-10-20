import onnxruntime as ort
import numpy as np

with open("./output.txt") as f:
    data = [l.strip() for l in f.readlines()]

outputs = {}

def parse_name(name):
    name, shape = name.split('(')
    return name, [int(t) for t in shape[:-1].split(",")]

for i in range(0, len(data), 2):
    if i + 1 >= len(data): break
    name = data[i]

    if name == "": break
    tdata = data[i + 1]

    print(name)
    name, shape = parse_name(name)
    outputs[name] = np.array([float(f) for f in tdata.split(" ")]).reshape(shape)

import onnx
from pathlib import Path

model_path = Path("~/Downloads/uc_merced_model(2).onnx").expanduser()
model = onnx.load(model_path)
# model.graph.output.extend(list(outputs.keys()))

# new_graph = onnx.helper.make_graph(model.graph.node, 'resnet', ['data'], list(outputs.keys()))

value_info_protos = []

inter_layers = list(outputs.keys())

shape_info = onnx.shape_inference.infer_shapes(model)
for idx, node in enumerate(shape_info.graph.value_info):
    if node.name in inter_layers:
        # print(idx, node)
        value_info_protos.append(node)

model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
onnx.checker.check_model(model)
onnx.save(model, './test.onnx')

onnx_input = model.graph.input[0]
input_shape = tuple(d.dim_value for d in onnx_input.type.tensor_type.shape.dim)
input = np.ones(input_shape, dtype=np.float32)

sess = ort.InferenceSession("./test.onnx")
ort_outputs = sess.run(list(outputs.keys()), {onnx_input.name: input}) 

print()
print("checking equality")
for (k, a), b in zip(outputs.items(), ort_outputs):
    if not np.equal(a, b).all():
        print(k)
        for f in a.flatten()[0:10]:
            print(f"{f:1.03f} ", end="")
        print()
        for f in b.flatten()[0:10]:
            print(f"{f:1.03f} ", end="")
        print()
        break
    else:
        print(f"{k} ✅")
