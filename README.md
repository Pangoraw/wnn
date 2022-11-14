# WNN - ONNX & WebGPU

## Simplifying

remove all i64 metadata from actual compute graph (they are not even available in wgpu).
and statically infer *everything* (this may require to export with very agressive inliner).

## TODOs:
 - [x] Infer Unet shapes/dtypes
 - [ ] Compile Unet to shaders
 - [ ] Support `f16`
 - [ ] Benchmark with tract.
