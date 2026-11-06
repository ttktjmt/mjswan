"""Writes the ONNX inputs in this directory with onnx's own serializer, so the cutters'
protobuf readers are tested against bytes neither wrote. Run from here with onnx and
numpy installed: `python make.py`. vectors.json records what each cuts to."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def tensor(name, count, value):
    """`count` float32s of `value`, stored as raw_data: a tensor of 4 * count bytes."""
    return numpy_helper.from_array(np.full(count, value, np.float32), name)


def constant(name, count, value):
    return helper.make_node("Constant", [], [name], value=tensor(name, count, value))


def model(nodes, initializers, metadata, producer="pytorch"):
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, "main", [], [out], initializers)
    m = helper.make_model(graph, producer_name=producer, opset_imports=[helper.make_opsetid("", 17)])
    m.doc_string = "test file for the blob cutter"
    for key, value in metadata.items():
        m.metadata_props.add(key=key, value=value)
    return m.SerializeToString()


def nested(depth, inner):
    """An If node whose then-branch nests `depth` graphs, the innermost holding `inner`."""
    branch = helper.make_graph(inner, f"g{depth}", [], [])
    for d in range(depth - 1, 0, -1):
        node = helper.make_node("If", ["cond"], [], then_branch=branch, else_branch=helper.make_graph([], "e", [], []))
        branch = helper.make_graph([node], f"g{d}", [], [])
    return helper.make_node("If", ["cond"], [], then_branch=branch, else_branch=helper.make_graph([], "e", [], []))


def xorshift32(seed, size):
    """The generated input of vectors.json: each step's state, four bytes little-endian."""
    x, out = seed, bytearray()
    while len(out) < size:
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out += x.to_bytes(4, "little")
    return bytes(out[:size])


def write(name, data):
    with open(name, "wb") as f:
        f.write(data)


# A policy: a Constant over 32 KiB, one in a subgraph, initializers on both sides of the
# bar (32,768 B is cut, 32,512 B is not), and metadata.
policy = model(
    [
        constant("scale", 10240, 1.0),
        helper.make_node(
            "If",
            ["cond"],
            ["branch"],
            then_branch=helper.make_graph([constant("then", 9216, 2.0)], "then", [], []),
            else_branch=helper.make_graph([constant("else", 16, 3.0)], "else", [], []),
        ),
    ],
    [
        tensor("obs_normalizer.mean", 100, 4.0),
        tensor("mlp.0.weight", 12800, 5.0),
        tensor("mlp.0.bias", 128, 6.0),
        tensor("mlp.2.weight", 8192, 7.0),
        tensor("mlp.2.bias", 64, 8.0),
        tensor("mlp.4.weight", 8128, 9.0),
        tensor("mlp.4.bias", 12, 10.0),
    ],
    {"run_path": "lab/g1-velocity/run-42", "joint_names": "hip,knee,ankle"},
)
write("policy.onnx", policy)

# Cut part-way through its graph: not read as protobuf, so not cut.
write("broken.onnx", policy[:50000])

# A term as mjswan writes one: small Constants and metadata, so one cut, before the metadata.
write(
    "term.onnx",
    model(
        [constant("threshold", 3, 0.5), constant("weights", 29, 1.5)],
        [],
        {"mjswan.kind": "termination", "mjswan.term": "fell_over"},
        producer="mjswan",
    ),
)

# 66 tensors at the bar and 3 of 256 KiB: 70 cuts, so the bar rises to a 64th of the file
# and keeps the 3 and the metadata.
write(
    "wide.onnx",
    model(
        [],
        [tensor(f"w{i}", 65536 if i in (10, 30, 50) else 8192, float(i)) for i in range(69)],
        {"note": "wide"},
    ),
)

# 63 tensors of 64 KiB: each still a 64th of the file, so 63 cuts and the metadata's, and
# the last is dropped.
write("crowded.onnx", model([], [tensor(f"w{i}", 16384, float(i)) for i in range(63)], {"note": "crowded"}))

# A Constant over the bar in the 16th graph down is cut; in the 17th, the file is not read.
write("deep16.onnx", model([nested(15, [constant("deep", 8192, 1.0)])], [], {}))
write("deep17.onnx", model([nested(16, [constant("deep", 8192, 1.0)])], [], {}))

# A tensor over 16 MiB, its piece cut again by content. Kept as the bytes around it; the
# tensor's own are generated (vectors.json: xorshift32, seed 2).
embed = xorshift32(2, 16 * 1024 * 1024 + 5)
big = model(
    [],
    [
        numpy_helper.from_array(np.frombuffer(embed, np.uint8), "embed"),
        tensor("head.weight", 10000, 1.0),
        tensor("head.bias", 12, 2.0),
    ],
    {"note": "big"},
)
at = big.find(embed[:64])
assert big[at : at + len(embed)] == embed
write("big-tensor.head", big[:at])
write("big-tensor.tail", big[at + len(embed) :])
