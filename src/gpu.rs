use std::collections::HashMap;

use anyhow::anyhow;
use wgpu::{include_wgsl, util::DeviceExt};

use crate::{onnx, shape::Shape, tensor::DataType};

pub(crate) struct TensorDesc {
    shape: Shape,
    dtype: DataType,
}

impl TensorDesc {
    pub(crate) fn new(shape: Shape, dtype: DataType) -> Self {
        Self { shape, dtype }
    }

    pub(crate) fn size_of(&self) -> usize {
        self.shape.numel().unwrap() * self.dtype.size_of()
    }
}

pub(crate) struct TensorStorage {
    desc: TensorDesc,
    buffer: wgpu::Buffer,
}

impl TensorStorage {
    pub(crate) fn new(
        device: &wgpu::Device,
        desc: TensorDesc,
        label: Option<&str>,
        is_output: bool,
    ) -> Self {
        let usage = if is_output {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            mapped_at_creation: false,
            size: desc.size_of() as _,
            usage,
        });
        Self { desc, buffer }
    }

    pub(crate) fn new_with_init(
        device: &wgpu::Device,
        data: &[u8],
        desc: TensorDesc,
        label: Option<&str>,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self { desc, buffer }
    }

    pub(crate) fn read_bytes(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.buffer.size(),
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &read_buf, 0, self.buffer.size());
        queue.submit(std::iter::once(encoder.finish()));

        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let out = slice.get_mapped_range().to_owned();
        read_buf.unmap();
        out
    }
}

/// A GPU operation to be run with inputs and outputs buffers.
pub(crate) struct Op {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch: (u32, u32, u32),
}

impl Op {
    pub(crate) fn new(
        device: &wgpu::Device,
        name: &str,
        inputs: Vec<&TensorStorage>,
        outputs: Vec<&TensorStorage>,
        op: &str,
    ) -> anyhow::Result<Self> {
        let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Shader {}", op)),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::from(std::fs::read_to_string(
                format!("shaders/{}.wgsl", op.to_lowercase()),
            )?)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Compute Pipeline {}", name)),
            layout: None,
            module: &kernel,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group Compute {}", name)),
            entries: &inputs
                .iter()
                .chain(outputs.iter())
                .enumerate()
                .map(|(i, tensor)| wgpu::BindGroupEntry {
                    binding: i as _,
                    resource: tensor.buffer.as_entire_binding(),
                })
                .collect::<Vec<wgpu::BindGroupEntry>>(),
            layout: &pipeline.get_bind_group_layout(0),
        });

        let (x, y) = match op {
            "MatMul" => (
                outputs[0].desc.shape.concrete_size(1)?,
                inputs[0].desc.shape.concrete_size(0)?,
            ),
            "Relu" => (inputs[0].desc.shape.numel().unwrap(), 1),
            _ => unimplemented!("{op}"),
        };

        Ok(Self {
            pipeline,
            bind_group,
            dispatch: (x as _, y as _, 1),
        })
    }

    pub(crate) fn run<'a>(
        &'a self,
        compute_pass: &mut wgpu::ComputePass<'a>,
    ) -> anyhow::Result<()> {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        let (x, y, z) = self.dispatch;
        compute_pass.dispatch_workgroups(x, y, z);

        Ok(())
    }
}

pub(crate) struct Runner<'a> {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    tensors: HashMap<&'a str, TensorStorage>,
}

impl<'a> Runner<'a> {
    pub(crate) async fn new() -> anyhow::Result<Runner<'a>> {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow!("failed to request adapter"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: Some("Compute Device"),
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            tensors: HashMap::new(),
        })
    }

    pub(crate) fn get_storage(&self, name: &str) -> &TensorStorage {
        &self.tensors[name]
    }

    pub(crate) fn add_node(
        &mut self,
        name: &'a str,
        desc: TensorDesc,
        is_output: bool,
    ) -> anyhow::Result<()> {
        let storage = TensorStorage::new(&self.device, desc, Some(name), is_output);
        if self.tensors.contains_key(name) {
            anyhow::bail!("node {} was already inserted in runner", name);
        }
        self.tensors.insert(name, storage);
        Ok(())
    }

    pub(crate) fn add_init(
        &mut self,
        tensor: &'a onnx::TensorProto,
        desc: TensorDesc,
    ) -> anyhow::Result<()> {
        self.add_node_with_init(tensor.name(), desc, tensor.raw_data())
    }

    pub(crate) fn add_node_with_init(
        &mut self,
        name: &'a str,
        desc: TensorDesc,
        raw_data: &[u8],
    ) -> anyhow::Result<()> {
        let storage = TensorStorage::new_with_init(&self.device, raw_data, desc, Some(name));
        if self.tensors.contains_key(name) {
            anyhow::bail!("tensor {} was already inserted in runner", name);
        }
        self.tensors.insert(name, storage);
        Ok(())
    }
}
