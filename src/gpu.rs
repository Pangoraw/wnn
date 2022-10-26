use std::collections::HashMap;

use anyhow::{anyhow, bail, Context};
use wgpu::util::DeviceExt;

use crate::{
    compiler::{compile_node, is_reshape_op, is_untracked_op},
    onnx,
    shape::Shape,
    tensor::DataType,
};

#[derive(Clone)]
pub(crate) struct TensorDesc {
    pub(crate) shape: Shape,
    pub(crate) dtype: DataType,
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
    pub(crate) fn size(&self) -> u64 {
        self.buffer.size()
    }

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
            size: self.size(),
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &read_buf, 0, self.size());
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
        inputs: Vec<&TensorStorage>,
        outputs: Vec<&TensorStorage>,
        node: &onnx::NodeProto,
        descs: &HashMap<&str, TensorDesc>,
    ) -> anyhow::Result<Self> {
        let shader = compile_node(node, descs)?;
        let shader_source = shader
            .to_wgsl()
            .with_context(|| anyhow!("compiling shader for {}", node.name()))?;

        // println!("{shader_source}");

        let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Shader {}", node.name())),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::from(shader_source)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Compute Pipeline {}", node.name())),
            layout: None,
            module: &kernel,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group Compute {}", node.name())),
            entries: &inputs
                .iter()
                .chain(&outputs)
                .enumerate()
                .map(|(i, tensor)| wgpu::BindGroupEntry {
                    binding: i as _,
                    resource: tensor.buffer.as_entire_binding(),
                })
                .collect::<Vec<wgpu::BindGroupEntry>>(),
            layout: &pipeline.get_bind_group_layout(0),
        });

        Ok(Self {
            pipeline,
            bind_group,
            dispatch: shader.dispatch(),
        })
    }

    pub(crate) fn run<'a>(&'a self, compute_pass: &mut wgpu::ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        let (x, y, z) = self.dispatch;
        compute_pass.dispatch_workgroups(x, y, z);
    }
}

pub(crate) struct Runner<'a> {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    tensors: HashMap<&'a str, TensorStorage>,

    slots: Vec<TensorStorage>,
    which_slots: HashMap<&'a str, usize>,
}

const MAX_ALLOC_LIMIT: u64 = 3_500_000_000;

impl<'a> Runner<'a> {
    pub(crate) async fn new() -> anyhow::Result<Runner<'a>> {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow!("failed to request adapter"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 268435456,
                        ..Default::default()
                    },
                    label: Some("Compute Device"),
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            tensors: HashMap::new(),
            slots: Vec::new(),
            which_slots: HashMap::new(),
        })
    }

    pub(crate) fn total_allocated_size(&self) -> u64 {
        self.tensors
            .values()
            .map(|tensor| tensor.size())
            .sum::<u64>()
            + self.slots.iter().map(|tensor| tensor.size()).sum::<u64>()
    }

    /// Performs tensor allocation by grouping allocations of the same size together.
    /// NOTE: We can probably do better by using bigger buffers to host smaller buffers.
    ///       but this seems good enough for models where the same activation size can often be seen.
    /// TODO: put inits, inputs and output buffers inside the allocation line too?
    pub(crate) fn allocate_tensors(
        &mut self,
        nodes: &'a [onnx::NodeProto],
        descs: &HashMap<&str, TensorDesc>,
        force_readable: bool,
    ) -> anyhow::Result<()> {
        let mut starts = HashMap::new();
        let mut ends = HashMap::new();

        let mut current_size = self.total_allocated_size();

        struct Slot {
            free: bool,
        }

        let mut slots: Vec<Slot> = Vec::new();

        // Dumb allocation strategy to be able to save the tensor activations
        if force_readable {
            for node in nodes {
                if is_reshape_op(node.op_type()) {
                    let input = &node.input[0];
                    let output = &node.output[0];

                    if self.tensors.contains_key(input.as_str()) {
                        self.tensors.remove(input.as_str());
                    }

                    if self.tensors.contains_key(output.as_str()) {
                        self.tensors.remove(input.as_str());
                    }

                    let input_desc = descs[input.as_str()].clone();
                    self.slots.push(TensorStorage::new(
                        &self.device,
                        input_desc,
                        None,
                        force_readable,
                    ));

                    slots.push(Slot { free: false });
                    self.which_slots.insert(input, slots.len() - 1);
                    self.which_slots.insert(output, slots.len() - 1);

                    continue;
                }

                for input in &node.input {
                    if self.tensors.contains_key(input.as_str())
                        || self.which_slots.contains_key(input.as_str())
                        || !descs.contains_key(input.as_str())
                    {
                        continue;
                    }
                    let desc = descs[input.as_str()].clone();
                    current_size += desc.size_of() as u64;
                    if current_size > MAX_ALLOC_LIMIT {
                        bail!(
                            "out-of-memory error when allocating {} (currently at {}) for {}",
                            human_bytes::human_bytes(desc.size_of() as f64),
                            human_bytes::human_bytes(current_size as f64),
                            input
                        );
                    }
                    self.add_node(input.as_str(), desc, true)?;
                }

                for output in &node.output {
                    if self.tensors.contains_key(output.as_str())
                        || self.which_slots.contains_key(output.as_str())
                        || !descs.contains_key(output.as_str())
                    {
                        continue;
                    }
                    let desc = descs[output.as_str()].clone();
                    current_size += desc.size_of() as u64;
                    if current_size > MAX_ALLOC_LIMIT {
                        bail!(
                            "out-of-memory error when allocating {} (currently at {}) for {}",
                            human_bytes::human_bytes(desc.size_of() as f64),
                            human_bytes::human_bytes(current_size as f64),
                            output
                        );
                    }
                    self.add_node(output.as_str(), desc, true)?;
                }
            }

            return Ok(());
        }

        let mut aliases = HashMap::new();
        for node in nodes {
            if is_reshape_op(node.op_type()) {
                let input = &node.input[0];
                let output = &node.output[0];

                aliases.insert(input, output);
            }
        }

        // Iterate the nodes in reverse to get liveness ranges for free.
        // See https://www.mattkeeter.com/blog/2022-10-04-ssra/ for more details.
        for (i, node) in nodes.iter().enumerate().rev() {
            if is_reshape_op(node.op_type()) || is_untracked_op(node.op_type()) {
                continue;
            }

            for input in node.input.iter().map(|input| match aliases.get(input) {
                Some(output) => output,
                None => input,
            }) {
                if self.tensors.contains_key(input.as_str()) || !descs.contains_key(input.as_str())
                {
                    continue;
                }

                // When node appears for the first time,
                // reserve a slot for it or allocate one.
                if !ends.contains_key(input) {
                    ends.insert(input, i);

                    // 2. Find free slot
                    if let Some((i, slot)) = slots.iter_mut().enumerate().find(|(i, slot)| {
                        slot.free
                            && self.slots[*i].desc.size_of() == descs[input.as_str()].size_of()
                    }) {
                        // 2.1) Reserve free slot if available
                        slot.free = false;
                        self.which_slots.insert(input.as_str(), i);
                    } else {
                        let desc = descs[input.as_str()].clone();

                        current_size += desc.size_of() as u64;
                        if current_size > MAX_ALLOC_LIMIT {
                            bail!(
                                "out-of-memory error when allocating {} (currently at {}) for {}",
                                human_bytes::human_bytes(desc.size_of() as f64),
                                human_bytes::human_bytes(current_size as f64),
                                input
                            );
                        }

                        // 2.2) Create slot otherwise
                        self.slots.push(TensorStorage::new(
                            &self.device,
                            desc,
                            None,
                            force_readable,
                        ));

                        slots.push(Slot { free: false });
                        self.which_slots.insert(input, slots.len() - 1);
                    }
                }
            }

            for output in node.output.iter().map(|input| match aliases.get(input) {
                Some(output) => output,
                None => input,
            }) {
                if self.tensors.contains_key(output.as_str())
                    || !descs.contains_key(output.as_str())
                {
                    continue;
                }

                starts.insert(output, i);

                // 1) When output appears mark slot as free
                let slot: &mut Slot = &mut slots[self.which_slots[output.as_str()]];
                slot.free = true;
            }
        }

        for (input, output) in aliases {
            assert!(!self.which_slots.contains_key(input.as_str()));
            self.which_slots
                .insert(input.as_str(), self.which_slots[output.as_str()]);
        }

        if &std::env::var_os("DUMP_ALLOCS")
            .map(|s| s.to_str().unwrap().to_owned())
            .unwrap_or_else(|| String::from("1"))
            == "1"
        {
            println!("=== ALLOCATIONS");
            for node in nodes {
                if is_untracked_op(node.op_type()) {
                    continue;
                }

                for (i, output) in node.output.iter().enumerate() {
                    print!(
                        "{}(#{})",
                        output,
                        match self.which_slots.get(output.as_str()) {
                            Some(s) => *s as isize,
                            None => -1,
                        }
                    );
                    if i != node.output.len() - 1 {
                        print!(", ");
                    }
                }
                print!("\t= {}[{}](", node.name(), node.op_type());
                let input_len = if node.op_type() == "Resize" {
                        1
                    } else {
                        node.input.len()
                    };
                for (i, input) in node
                    .input
                    .iter()
                    .take(input_len)
                    .enumerate()
                {
                    print!(
                        "{}(#{})",
                        input,
                        match self.which_slots.get(input.as_str()) {
                            Some(s) => *s as isize,
                            None => -1,
                        }
                    );
                    if i != input_len - 1 {
                        print!(", ");
                    }
                }
                println!(")")
            }
            println!("===============");
        }

        Ok(())
    }

    pub(crate) fn get_storage(&self, name: &str) -> anyhow::Result<&TensorStorage> {
        self.tensors
            .get(name)
            .or_else(|| match self.which_slots.get(name) {
                Some(i) if *i < self.slots.len() => Some(&self.slots[*i]),
                _ => None,
            })
            .ok_or_else(|| anyhow!("tensor {} not found", name))
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
        let raw_data = if matches!(desc.dtype, DataType::F32) && !tensor.float_data.is_empty() {
            bytemuck::cast_slice(&tensor.float_data)
        } else if matches!(desc.dtype, DataType::I64) && !tensor.int64_data.is_empty() {
            bytemuck::cast_slice(&tensor.int64_data)
        } else {
            tensor.raw_data()
        };
        self.add_node_with_init(tensor.name(), desc, raw_data)
    }

    pub(crate) fn add_node_with_init(
        &mut self,
        name: &'a str,
        desc: TensorDesc,
        raw_data: &[u8],
    ) -> anyhow::Result<()> {
        if raw_data.len() != desc.size_of() {
            bail!(
                "invalid raw_data (len {}) for tensor {} of size {}, type {}",
                raw_data.len(),
                name,
                desc.shape,
                desc.dtype,
            );
        }
        let storage = TensorStorage::new_with_init(&self.device, raw_data, desc, Some(name));
        if self.tensors.contains_key(name) {
            anyhow::bail!("tensor {} was already inserted in runner", name);
        }
        self.tensors.insert(name, storage);
        Ok(())
    }
}
