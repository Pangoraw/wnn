# WNN - ONNX & WebGPU

> [!note]
>
> Everything that this repository implemented (shape inference, buffer
> allocation) is included in [webonnx](https://github.com/webonnx).

## Ideas for speed

ONNX runtime just ships the ONNX runtime and ONNX files for inference which can
take a while to download on lighter bandwith. The difference here would be
between compiled and interpreted languages.

- Break model in multiple sections, load asynchronously while performing
  inference on the first part of the network. The goal here is reducing time to
  first inference.
- Format for JavaScript only inference with simple commands, (i.e. copy buffer,
  launch shader, and so on) with minimal runtime.

Provide Transformers.js-like _compile-time_ API. Web tools already have build
steps, why not have one on the most weight heavy component.
