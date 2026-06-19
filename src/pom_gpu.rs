//! Proof-of-Model GPU mining — runs the `pom_mine` kernel in candle's CUDA context over the
//! resident weight blob to find a winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! Loads the mining tier's GGUF raw (so we get per-tensor device pointers for the gather, like
//! `pom-q4-probe`) and builds the chunk-prefix gather index on the GPU. NOTE: this is a second
//! VRAM copy of the model (the inference engine holds its own). Fine for small tiers on the
//! testnet; the big tiers will share buffers later.
//!
//! The kernel's seed/pow folds are byte-identical to `pom::pom_block_seed`/`pom::pom_pow_value`,
//! so a nonce found here builds a `PomProof` (host) the node accepts.

use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));
const CHUNK_BYTES: usize = 32;

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

pub struct PomGpuMiner {
    cuda: CudaDevice,
    stream: Arc<CudaStream>,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u64>,
    t_count: u32,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>, // kept alive so the gather pointers stay valid (resident in VRAM)
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into candle (device 0), build the gather index, load the kernel.
    pub fn load(gguf_path: &str) -> candle_core::Result<Self> {
        let device = Device::new_cuda(0)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — matches pom-rt-builder / the node R_T

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            let chunks = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                tensors.push(qt);
                continue;
            }
            bases.push(qt.device_ptr()? as usize as u64);
            prefix.push(prefix.last().unwrap() + chunks);
            tensors.push(qt);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }

        let bases_dev = stream.memcpy_stod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.memcpy_stod(&prefix).map_err(candle_core::Error::wrap)?;
        // Warm the module cache so mine() never compiles on the hot path.
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: tensors })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> candle_core::Result<Option<u64>> {
        let p = words4(pre_pow_hash);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = self.stream.memcpy_stod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        let func = self.cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?; // cached
        let mut b = func.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&self.n_total_chunks).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3]).arg(&timestamp)
            .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
            .arg(&start).arg(&batch).arg(&winner);
        unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.stream.memcpy_dtov(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}
