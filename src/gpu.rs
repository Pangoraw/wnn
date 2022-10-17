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
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ
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

    pub(crate) fn to_bytes(&self, device: &wgpu::Device) -> Vec<u8> {
        let slice = self.buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let out = slice.get_mapped_range().to_owned();
        self.buffer.unmap();
        out
    }
}

pub(crate) struct Op {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl Op {
    pub(crate) fn new(
        device: &wgpu::Device,
        name: &str,
        inputs: Vec<&TensorStorage>,
        outputs: Vec<&TensorStorage>,
    ) -> Self {
        let kernel = device.create_shader_module(include_wgsl!("kernel.wgsl"));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Compute Pipeline {}", name)),
            layout: None,
            module: &kernel,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Compute"),
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

        Self {
            pipeline,
            bind_group,
        }
    }

    pub(crate) fn run<'a>(
        &'a self,
        compute_pass: &mut wgpu::ComputePass<'a>,
    ) -> anyhow::Result<()> {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 2, 1);

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
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow!("failed to request adapter"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
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
        println!(
            "name = {}, {:?}",
            name,
            self.tensors.keys().collect::<Vec<&&str>>()
        );
        return &self.tensors[name];
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
