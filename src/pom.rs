//! Proof-of-Model — miner-side possession proof builder (build order §6).
//!
//! Byte-exact mirror of the node's verifier (`keryx-node-hardfork consensus/core/src/pom.rs`)
//! and the canonical reference (`pom-core`). The miner runs the memory-hard walk over its
//! resident weight blob; once a winning nonce is found, `build_proof` re-walks (recording the
//! trace), commits it, and opens the `t` Fiat-Shamir-selected steps with Merkle paths to the
//! tier root `R_T` and the trace root.
//!
//! The `PomProof`/`PomOpening` structs MUST keep the exact field order/types of the node's
//! (borsh wire format), and the primitives MUST stay bit-identical (the node re-derives the
//! same challenges and recomputes the same transitions). See POM_CONSENSUS_SPEC.md.

use borsh::{BorshDeserialize, BorshSerialize};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::sync::OnceLock;

pub const CHUNK_WORDS: usize = 4; // 32 B chunk
const SEED_SALT: u64 = 0x4B65727978500; // "KeryxP"

/// Walk length / opening count — MUST match the node's `POM_WALK_STEPS` / `POM_OPENINGS`.
/// K=256 — chosen compromise (~25 MH/s on a 3090, solid possession).
pub const POM_WALK_STEPS: u32 = 256;
pub const POM_OPENINGS: usize = 32;

// --- wire structs (field order == node's PomOpening/PomProof) ---

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomOpening {
    pub state_before: u64,
    pub chunk: [u8; 32],
    pub weight_path: Vec<[u8; 32]>,
    pub trace_path_before: Vec<[u8; 32]>,
    pub trace_path_after: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProof {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
}

// --- byte-exact primitives (mirror node) ---

#[inline]
pub fn blake(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline]
pub fn seed_state(pow_seed: u64) -> u64 {
    mix64(pow_seed ^ SEED_SALT)
}

#[inline]
pub fn transition(state: u64, chunk: &[u64; CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
}

#[inline]
pub fn chunk_to_words(c: &[u8; 32]) -> [u64; CHUNK_WORDS] {
    let mut w = [0u64; CHUNK_WORDS];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[inline]
pub fn words_to_bytes(w: &[u64; CHUNK_WORDS]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, wi) in w.iter().enumerate() {
        b[i * 8..i * 8 + 8].copy_from_slice(&wi.to_le_bytes());
    }
    b
}

#[inline]
fn trace_leaf(state: u64) -> [u8; 32] {
    blake(&state.to_le_bytes())
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake(&buf)
}

pub fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

#[inline]
fn pph_words(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(pre_pow_hash[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Canonical block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_seed_fold` and the node's `pom_block_seed`.
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    let p = pph_words(pre_pow_hash);
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical pow value (256-bit LE) = mix64-fold of (final_state, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_pow_fold` and the node's `pom_pow_value`.
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32]) -> [u8; 32] {
    let p = pph_words(pre_pow_hash);
    let o0 = mix64(final_state ^ p[0] ^ 0x9E3779B97F4A7C15);
    let o1 = mix64(o0 ^ p[1] ^ 0xC2B2AE3D27D4EB4F);
    let o2 = mix64(o1 ^ p[2] ^ 0x165667B19E3779F9);
    let o3 = mix64(o2 ^ p[3] ^ 0xD6E8FEB86659FD93);
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&o0.to_le_bytes());
    out[8..16].copy_from_slice(&o1.to_le_bytes());
    out[16..24].copy_from_slice(&o2.to_le_bytes());
    out[24..32].copy_from_slice(&o3.to_le_bytes());
    out
}

pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "merkle_root: empty leaves");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
        let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
        path.push(sib);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        idx >>= 1;
        level = next;
    }
    path
}

fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices — byte-layout identical to node/pom-core.
pub fn challenges(pre_pow_hash: &[u8; 32], nonce: u64, trace_root: &[u8; 32], pow_value: &[u8; 32], t: usize, k: u32) -> Vec<u32> {
    let mut fs = [0u8; 104];
    fs[..32].copy_from_slice(pre_pow_hash);
    fs[32..40].copy_from_slice(&nonce.to_le_bytes());
    fs[40..72].copy_from_slice(trace_root);
    fs[72..104].copy_from_slice(pow_value);
    let seed = blake(&fs);
    let mut out = Vec::with_capacity(t);
    for j in 0..t as u64 {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&j.to_le_bytes());
        let d = blake(&buf);
        let v = u64::from_le_bytes(d[..8].try_into().unwrap());
        out.push((v % k as u64) as u32);
    }
    out
}

/// The hot search walk: K data-dependent reads, returns only `state[K]` (no trace recording).
/// This is the per-nonce work; on GPU (slice 3b) this becomes the kernel over VRAM weights.
pub fn walk_final<F: Fn(u64) -> [u64; CHUNK_WORDS]>(seed: u64, n_chunks: u64, k: u32, read_chunk: F) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        off = state % n_chunks;
    }
    state
}

/// CPU Proof-of-Model mining (slice 3a — functional, slow). Searches nonces in
/// `nonce_start..nonce_start+max_nonces`; on the first whose `pom_pow_value <= target`,
/// re-walks to build the full `PomProof`. GPU fast-path is slice 3b. Returns the winning
/// nonce + proof, or None if the range is exhausted.
#[allow(clippy::too_many_arguments)]
pub fn mine_pom(
    index: &WeightIndex,
    tier: u8,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target: &[u8; 32],
    k: u32,
    t: usize,
    nonce_start: u64,
    max_nonces: u64,
) -> Option<(u64, PomProof)> {
    for nonce in nonce_start..nonce_start.saturating_add(max_nonces) {
        let seed = pom_block_seed(pre_pow_hash, timestamp, nonce);
        let final_state = walk_final(seed, index.n_chunks, k, |o| index.read_chunk(o));
        if le_leq(&pom_pow_value(final_state, pre_pow_hash), target) {
            let proof = build_proof(tier, pre_pow_hash, nonce, seed, index.n_chunks, k, t, |o| index.read_chunk(o), |o| index.merkle_path(o));
            return Some((nonce, proof));
        }
    }
    None
}

/// PROVER. Re-walk the (already-won) nonce recording the trace, commit it, and open the
/// `t` FS-selected steps. `read_chunk(off)` reads the 32 B chunk at canonical chunk index
/// `off` from the resident weight blob; `weight_leaves` is the precomputed per-chunk leaf
/// set (`blake(chunk_bytes)`) over the canonical layout, used to produce weight Merkle paths.
#[allow(clippy::too_many_arguments)]
pub fn build_proof<F, WP>(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    n_chunks: u64,
    k: u32,
    t: usize,
    read_chunk: F,
    weight_path: WP,
) -> PomProof
where
    F: Fn(u64) -> [u64; CHUNK_WORDS],
    WP: Fn(u64) -> Vec<[u8; 32]>,
{
    let mut trace = Vec::with_capacity(k as usize + 1);
    let mut state = seed;
    trace.push(state);
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        trace.push(state);
        off = state % n_chunks;
    }
    let trace_leaves: Vec<[u8; 32]> = trace.iter().map(|&s| trace_leaf(s)).collect();
    let trace_root = merkle_root(&trace_leaves);
    let final_state = trace[k as usize];
    let pow_value = pom_pow_value(final_state, pre_pow_hash);

    let chs = challenges(pre_pow_hash, nonce, &trace_root, &pow_value, t, k);
    let openings = chs
        .iter()
        .map(|&i| {
            let i = i as usize;
            let sb = trace[i];
            let off = sb % n_chunks;
            PomOpening {
                state_before: sb,
                chunk: words_to_bytes(&read_chunk(off)),
                weight_path: weight_path(off),
                trace_path_before: merkle_proof(&trace_leaves, i),
                trace_path_after: merkle_proof(&trace_leaves, i + 1),
            }
        })
        .collect();

    PomProof {
        tier,
        trace_root,
        pow_value,
        final_state,
        initial_trace_path: merkle_proof(&trace_leaves, 0),
        final_trace_path: merkle_proof(&trace_leaves, k as usize),
        openings,
    }
}

/// Self-check a built proof before submit (same logic the node runs). Cheap insurance
/// against emitting a block the node will reject.
#[allow(clippy::too_many_arguments)]
pub fn verify_proof(pre_pow_hash: &[u8; 32], nonce: u64, seed: u64, proof: &PomProof, n_chunks: u64, k: u32, t: usize, r_t: &[u8; 32], target: &[u8; 32]) -> bool {
    if proof.openings.len() != t {
        return false;
    }
    if pom_pow_value(proof.final_state, pre_pow_hash) != proof.pow_value {
        return false;
    }
    if !le_leq(&proof.pow_value, target) {
        return false;
    }
    if !verify_merkle(trace_leaf(seed), 0, &proof.initial_trace_path, &proof.trace_root) {
        return false;
    }
    if !verify_merkle(trace_leaf(proof.final_state), k as u64, &proof.final_trace_path, &proof.trace_root) {
        return false;
    }
    let chs = challenges(pre_pow_hash, nonce, &proof.trace_root, &proof.pow_value, t, k);
    for (op, &i) in proof.openings.iter().zip(chs.iter()) {
        let i = i as u64;
        if !verify_merkle(trace_leaf(op.state_before), i, &op.trace_path_before, &proof.trace_root) {
            return false;
        }
        let off = op.state_before % n_chunks;
        if !verify_merkle(blake(&op.chunk), off, &op.weight_path, r_t) {
            return false;
        }
        let state_after = transition(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return false;
        }
    }
    true
}

/// Canonical weight index built once at startup from the resident model: the per-chunk
/// blake3 leaves (for Merkle paths), the recomputed tier root `R_T` (sanity-checked against
/// the consensus-pinned value), and a chunk reader. Canonical layout = name-sorted GGUF
/// tensors, `floor(len/32)` 32 B chunks — identical to `pom-rt-builder` and the node.
///
/// NOTE: holds the canonical chunk bytes + leaves in host RAM (~2x model size). Fine for
/// the small/mid tiers; the big tiers will switch to reading chunks straight from the
/// resident VRAM buffers (slice 3) instead of a host copy.
pub struct WeightIndex {
    pub n_chunks: u64,
    pub r_t: [u8; 32],
    /// Full Merkle tree, level 0 = leaves up to the single-node root level. Stored so each
    /// `merkle_path` is O(log N) instead of rebuilding the tree per call (proofs are built at
    /// block frequency — rebuilding ~N hashes per path would be unusable).
    tree: Vec<Vec<[u8; 32]>>,
    data: Vec<u8>,
}

impl WeightIndex {
    /// Build from a GGUF on disk (CPU dtoh of each tensor). The bytes are candle's exact
    /// quantized bytes — the same the miner serves in VRAM and the builder pinned in `R_T`.
    pub fn build_from_gguf(path: &str) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let mut file = std::fs::File::open(path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order

        let mut leaves: Vec<[u8; 32]> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            let bytes = qt.data()?;
            let full = bytes.len() / 32;
            for c in 0..full {
                let chunk = &bytes[c * 32..c * 32 + 32];
                leaves.push(blake(chunk));
                data.extend_from_slice(chunk);
            }
        }
        if leaves.is_empty() {
            return Err(candle_core::Error::Msg("PoM: model produced 0 chunks".into()));
        }
        let n_chunks = leaves.len() as u64;

        // Build all tree levels once (duplicate-last on odd levels — matches merkle_root).
        let mut tree: Vec<Vec<[u8; 32]>> = vec![leaves];
        while tree.last().unwrap().len() > 1 {
            let level = tree.last().unwrap();
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i < level.len() {
                let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
                next.push(hash_pair(&level[i], &r));
                i += 2;
            }
            tree.push(next);
        }
        let r_t = tree.last().unwrap()[0];
        Ok(Self { n_chunks, r_t, tree, data })
    }

    /// 32 B chunk at canonical index `off` (panics if out of range — `off < n_chunks`).
    pub fn read_chunk(&self, off: u64) -> [u64; CHUNK_WORDS] {
        let base = (off as usize) * 32;
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&self.data[base..base + 32]);
        chunk_to_words(&arr)
    }

    /// Inclusion path for chunk index `off` from the prebuilt tree — O(log N).
    pub fn merkle_path(&self, off: u64) -> Vec<[u8; 32]> {
        let mut path = Vec::with_capacity(self.tree.len());
        let mut idx = off as usize;
        for level in &self.tree[..self.tree.len() - 1] {
            let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
            let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
            path.push(sib);
            idx >>= 1;
        }
        path
    }
}

/// PoM possession activation DAA score — MUST match the node's `pom_activation`.
/// `u64::MAX` = never (dormant): mining stays on legacy kHeavyHash, no proof produced.
///
/// ⚠️ TESTNET VALUE — DO NOT COMMIT. `1000` = activate together with OPoI v2 (one GPU→GPU
/// cutover, no difficulty cliff). Mainnet: H. Revert to `u64::MAX` before release.
pub const POM_ACTIVATION_DAA: u64 = 1000;

/// The resident tier weight index + tier id, installed once at startup when PoM is enabled.
static POM_INDEX: OnceLock<(WeightIndex, u8)> = OnceLock::new();

/// Install the possession index (built from the resident model) and its tier. Call once.
pub fn set_index(index: WeightIndex, tier: u8) {
    let _ = POM_INDEX.set((index, tier));
}

/// The active possession index + tier, if installed.
pub fn active_index() -> Option<&'static (WeightIndex, u8)> {
    POM_INDEX.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_chunk(off: u64) -> [u64; CHUNK_WORDS] {
        let mut c = [0u64; CHUNK_WORDS];
        for (j, w) in c.iter_mut().enumerate() {
            *w = mix64(off.wrapping_mul(CHUNK_WORDS as u64) + j as u64 + 1);
        }
        c
    }

    // Synthetic WeightIndex (no GGUF) — exercises the real read_chunk + O(log N) merkle_path.
    fn synth_index(n: u64) -> WeightIndex {
        let mut leaves = Vec::new();
        let mut data = Vec::new();
        for o in 0..n {
            let b = words_to_bytes(&synth_chunk(o));
            leaves.push(blake(&b));
            data.extend_from_slice(&b);
        }
        let mut tree = vec![leaves];
        while tree.last().unwrap().len() > 1 {
            let level = tree.last().unwrap();
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i < level.len() {
                let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
                next.push(hash_pair(&level[i], &r));
                i += 2;
            }
            tree.push(next);
        }
        let r_t = tree.last().unwrap()[0];
        WeightIndex { n_chunks: n, r_t, tree, data }
    }

    #[test]
    fn weight_index_root_matches_standalone() {
        // The prebuilt-tree root equals the standalone merkle_root over the same leaves.
        let n = 1000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        assert_eq!(idx.r_t, merkle_root(&leaves));
    }

    #[test]
    fn build_then_self_verify() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph");
        let nonce = 0xabc;
        let seed = pom_block_seed(&pph, 111, nonce);

        let proof = build_proof(2, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32]));
        // borsh wire-format round-trips (same encoding the node decodes).
        let bytes = borsh::to_vec(&proof).unwrap();
        let back: PomProof = borsh::from_slice(&bytes).unwrap();
        assert!(verify_proof(&pph, nonce, seed, &back, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32]));
        assert_eq!(back.tier, 2);
    }

    #[test]
    fn wrong_target_or_root_fails() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph2");
        let nonce = 7;
        let seed = pom_block_seed(&pph, 1, nonce);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(!verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0u8; 32]), "zero target must fail");
        assert!(!verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &blake(b"wrong"), &[0xff; 32]), "wrong R_T must fail");
    }

    #[test]
    fn cpu_mine_finds_nonce_and_proof_verifies() {
        let (k, t) = (128u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"mine-pph");
        let ts = 555;
        // Target requiring pow_value MSB <= 0x10 (~6.6% of nonces) — found within a few tries.
        let mut target = [0xffu8; 32];
        target[31] = 0x10;
        let (nonce, proof) = mine_pom(&idx, 1, &pph, ts, &target, k, t, 0, 100_000).expect("mine a nonce");
        let seed = pom_block_seed(&pph, ts, nonce);
        // The proof verifies against the same target the node would use.
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target));
        assert_eq!(proof.tier, 1);
    }

    // Validates the canonical layout against the consensus-pinned R_T. Needs the Gemma GGUF.
    // Run: cargo test --lib pom -- --ignored --nocapture
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk"]
    fn weight_index_matches_pinned_gemma() {
        let path = "/home/slash/KERYX-KRX/claude/keryx-miner/target/release/models/Gemma-3-4B/model.gguf";
        let idx = WeightIndex::build_from_gguf(path).expect("build index");
        assert_eq!(idx.n_chunks, 77_604_776, "chunk count must match pinned GEMMA_3_4B_POM_CHUNKS");
        let pinned: [u8; 32] = [
            0x84, 0x6c, 0xaa, 0x40, 0x0c, 0xf0, 0x14, 0x13, 0x21, 0x18, 0x49, 0x5d, 0x22, 0xe4, 0xbf, 0xa2,
            0x42, 0x45, 0x4e, 0xac, 0x0d, 0x83, 0x5c, 0x3f, 0x8e, 0x63, 0x47, 0xd0, 0x13, 0x9d, 0x1b, 0x7e,
        ];
        assert_eq!(idx.r_t, pinned, "miner R_T must equal node-pinned GEMMA_3_4B_POM_ROOT");

        // A real proof over the real model self-verifies against the pinned R_T.
        let pph = blake(b"gemma-pph");
        let nonce = 1234;
        let seed = pom_block_seed(&pph, 99, nonce);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, 256, 32, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, 256, 32, &idx.r_t, &[0xff; 32]));
    }
}
