#!/bin/bash

wnn() {
    RUST_LOG=wnn cargo run -- $@
}

rm activations/*
 wnn ../ONNX.jl/model_sim.onnx \
    --init ~/irisa/diffusers/latents_ninja.npy

# ../ONNX.jl/model_512.onnx # ~/irisa/diffusers/decoder_v1_4_pytorch_sim.onnx # ../ONNX.jl/model_sim.onnx # 
python ./preview_result.py
