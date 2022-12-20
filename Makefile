test: wnn-wasm/resnet.onnx
	cargo test

wnn-wasm/resnet.onnx:
	wget https://webml.netlify.app/resnet.onnx; mv resnet.onnx wnn-wasm/
