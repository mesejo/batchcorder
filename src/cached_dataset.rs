//! Cached Arrow dataset backed by an in-memory Vec or an on-disk IPC file.
//!
//! # Overview
//!
//! [`PyStreamCache`] accepts any upstream Arrow stream source (anything that
//! exposes `__arrow_c_stream__` in Python) and stores each `RecordBatch` in a
//! cache keyed by a monotonic `u64` batch index.  Two storage modes:
//!
//! - **Memory-only** (`disk_path` / `disk_capacity` omitted): batches are kept
//!   as `Arc<RecordBatch>` in a `Vec`.  Reads are zero-copy Arc clones; no IPC
//!   serialisation happens at all.
//! - **Disk** (`disk_path` + `disk_capacity` both provided): batches are
//!   serialised to Arrow IPC stream format and appended to a single temp file.
//!   A hot in-memory layer (`memory_capacity` bytes) avoids disk reads for
//!   recently ingested batches; entries that exceed the hot budget are read
//!   back from disk via positional I/O (no seek lock needed).
//!
//! Multiple independent [`PyStreamCacheReader`] handles can be obtained from a
//! single dataset, each maintaining its own read position.

use std::collections::VecDeque;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

// ── system memory detection ───────────────────────────────────────────────────

static TOTAL_SYSTEM_MEMORY: LazyLock<usize> = LazyLock::new(|| {
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    sys.total_memory() as usize
});

use arrow_array::RecordBatch;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_pyarrow::{IntoPyArrow, PyArrowType};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use pyo3::exceptions::{PyIOError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_stub_gen::derive::*;
use std::ffi::CString;
use xxhash_rust::xxh3::xxh3_64;

// ── dataset counter ───────────────────────────────────────────────────────────

static DATASET_COUNTER: AtomicU64 = AtomicU64::new(0);

// ── write policy ───────────────────────────────────────────────────────────────

/// Controls when batches are written to the disk tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WritePolicy {
    /// Write every batch to disk immediately on insertion.  The hot layer is a
    /// read-only cache; disk is always the source of truth.  Lower memory use,
    /// more I/O.
    OnInsertion,
    /// Write a batch to disk only when it is evicted from the hot layer.  If the
    /// hot layer is large enough, nothing ever hits disk.  Higher memory use for
    /// hot entries; far less I/O for small-to-medium streams.
    OnEviction,
}

// ── cache tiers ──────────────────────────────────────────────────────────────

/// In-memory cache: batches stored as `Arc<RecordBatch>`, zero-copy reads.
struct MemoryCacheTier {
    batches: RwLock<Vec<Arc<RecordBatch>>>,
    /// Byte budget for all cached batches combined.
    capacity: usize,
    /// Bytes currently held across all cached batches.
    used: AtomicUsize,
}

/// Disk-backed write state (mutated only during ingestion).
struct DiskWriteState {
    file: std::fs::File,
    offset: u64,
}

/// FIFO in-memory read cache for the disk tier.
///
/// Entries are keyed by a monotonic batch index.  When the byte budget is
/// exceeded the oldest entry is evicted first — optimal for sequential replay
/// because batch N is always read before batch N+1.
struct HotLayer {
    /// `entries[0]` corresponds to batch `head_batch_idx`.
    entries: VecDeque<Option<(Arc<RecordBatch>, usize)>>,
    /// Global batch index of `entries[0]`.
    head_batch_idx: u64,
    /// Bytes currently held (sum of ipc_byte_len across all `Some` entries).
    used: usize,
    /// Byte budget.
    capacity: usize,
}

impl HotLayer {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            head_batch_idx: 0,
            used: 0,
            capacity,
        }
    }

    /// Insert `batch` at `batch_idx`, evicting from the front until there is room.
    /// Returns `false` when the batch is larger than the entire budget (not cached,
    /// but callers must still write it to disk — returning false is not an error).
    fn try_insert(&mut self, batch_idx: u64, batch: Arc<RecordBatch>, ipc_len: usize) -> bool {
        while self.used + ipc_len > self.capacity {
            match self.entries.pop_front() {
                None => {
                    // The batch exceeds the whole budget, so it is not cached.
                    // It still consumes a global batch index, so advance head
                    // past it to keep `entries[0] == head_batch_idx` (the
                    // mapping `get` relies on).  Without this, a later batch
                    // that *does* fit would be stored at — and read back under —
                    // the wrong index when batch sizes vary.
                    self.head_batch_idx += 1;
                    return false;
                }
                Some(evicted) => {
                    if let Some((_, len)) = evicted {
                        self.used -= len;
                    }
                    self.head_batch_idx += 1;
                }
            }
        }
        debug_assert_eq!(batch_idx, self.head_batch_idx + self.entries.len() as u64);
        self.entries.push_back(Some((batch, ipc_len)));
        self.used += ipc_len;
        true
    }

    /// Add `batch` to the back and return the front entries that must be evicted
    /// to make room, **without removing them yet**.  The returned entries stay in
    /// the hot layer (accessible to concurrent readers) until [`Self::commit_evictions`]
    /// is called — this is the two-phase commit that closes the window where a
    /// batch would be neither in hot nor on disk.
    ///
    /// Used by the `OnEviction` policy: the caller writes each returned batch to
    /// disk, then calls `commit_evictions` with the returned count.
    fn stage_evictions(
        &mut self,
        batch_idx: u64,
        batch: Arc<RecordBatch>,
        ipc_len: usize,
    ) -> Vec<(u64, Arc<RecordBatch>, usize)> {
        let mut evicted = Vec::new();
        let mut cursor = 0usize;
        let mut freed = 0usize;
        // Collect just enough front entries to fit the newcomer; do not pop them.
        while self.used - freed + ipc_len > self.capacity {
            match self.entries.get(cursor) {
                None => break, // nothing left to evict
                Some(slot) => {
                    if let Some((arc, len)) = slot {
                        evicted.push((self.head_batch_idx + cursor as u64, arc.clone(), *len));
                        freed += len;
                    }
                    cursor += 1;
                }
            }
        }
        debug_assert_eq!(batch_idx, self.head_batch_idx + self.entries.len() as u64);
        let _ = batch_idx;
        self.entries.push_back(Some((batch, ipc_len)));
        self.used += ipc_len;
        evicted
    }

    /// Pop the first `n` entries after their disk writes have been confirmed.
    /// `n` must equal the length returned by the matching [`Self::stage_evictions`].
    fn commit_evictions(&mut self, n: usize) {
        for _ in 0..n {
            match self.entries.pop_front() {
                None => break,
                Some(evicted) => {
                    if let Some((_, len)) = evicted {
                        self.used -= len;
                    }
                    self.head_batch_idx += 1;
                }
            }
        }
    }

    fn get(&self, batch_idx: u64) -> Option<Arc<RecordBatch>> {
        if batch_idx < self.head_batch_idx {
            return None;
        }
        let slot = (batch_idx - self.head_batch_idx) as usize;
        self.entries.get(slot)?.as_ref().map(|(arc, _)| arc.clone())
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }
}

/// Per-batch index record for the disk tier.
#[derive(Copy, Clone)]
struct DiskEntry {
    /// Byte offset of `[checksum_header | ipc_payload]` in the cache file.
    file_offset: u64,
    /// IPC payload length in bytes (excludes the 8-byte checksum header).
    ipc_len: usize,
    /// xxh3_64 of the IPC payload, cached in RAM to avoid re-reading the header.
    checksum: u64,
}

/// Combined index + hot layer for the disk tier.
struct DiskIndex {
    /// `entries[i]` is `Some` once batch `i` is on disk, `None` while it lives only
    /// in the hot layer (the `OnEviction` policy before eviction).  Under
    /// `OnInsertion` every slot is `Some` immediately.
    entries: Vec<Option<DiskEntry>>,
    hot: HotLayer,
}

/// Disk-backed cache: append-only IPC file + optional hot in-memory layer.
struct DiskCacheTier {
    /// Subdirectory created under the caller's `disk_path`; removed on cleanup.
    dir_path: PathBuf,
    /// Serialised write position (locked separately; never contended — ingestion
    /// is always serialised by the outer `DatasetInner` mutex).
    write_state: Mutex<DiskWriteState>,
    /// Batch locations and hot-cache entries (RwLock: many concurrent readers,
    /// one writer at a time during ingestion).
    index: RwLock<DiskIndex>,
    /// File descriptor used for positional reads (pread-style, no seek lock).
    read_file: std::fs::File,
    /// Hard byte limit on the on-disk cache file.
    disk_capacity: u64,
    /// Bytes written to disk so far.  Relaxed ordering is safe: single writer
    /// (serialised by `DatasetInner` mutex), readers never inspect this field.
    disk_used: AtomicU64,
    /// When batches are flushed to disk (immediately, or on hot eviction).
    policy: WritePolicy,
}

impl DiskCacheTier {
    /// Fail unless `8 + length` more bytes fit within `disk_capacity`.
    fn check_disk_capacity(&self, length: usize) -> Result<(), ArrowError> {
        let prev = self.disk_used.load(Ordering::Relaxed);
        if prev
            .checked_add((8 + length) as u64)
            .is_none_or(|end| end > self.disk_capacity)
        {
            return Err(ArrowError::MemoryError(format!(
                "Disk cache capacity ({} bytes) exceeded: {} bytes already written, \
                 cannot fit {} more bytes",
                self.disk_capacity,
                prev,
                8 + length
            )));
        }
        Ok(())
    }

    /// Append `[checksum_le | bytes]` to the cache file and return the start
    /// offset.  Advances `disk_used` and the write offset.  Caller must have
    /// already passed [`Self::check_disk_capacity`].
    fn write_ipc_to_disk(&self, bytes: &[u8], checksum: u64) -> Result<u64, ArrowError> {
        let length = bytes.len();
        let offset = {
            let mut ws = self.write_state.lock().map_err(poisoned_lock_err)?;
            let off = ws.offset;
            let mut on_disk = Vec::with_capacity(8 + length);
            on_disk.extend_from_slice(&checksum.to_le_bytes());
            on_disk.extend_from_slice(bytes);
            ws.file
                .write_all(&on_disk)
                .map_err(|e| ArrowError::IoError(format!("Disk write failed: {e}"), e))?;
            // Flush so subsequent pread calls on read_file see the bytes
            // (the kernel buffer cache is shared between the two FDs).
            ws.file
                .flush()
                .map_err(|e| ArrowError::IoError(format!("Disk flush failed: {e}"), e))?;
            ws.offset = ws
                .offset
                .checked_add((8 + length) as u64)
                .ok_or_else(|| other_arrow_err("Cache file offset overflowed"))?;
            off
        };
        self.disk_used
            .fetch_add((8 + length) as u64, Ordering::Relaxed);
        Ok(offset)
    }

    /// `OnInsertion`: serialise and write the batch immediately, then record it
    /// in the hot layer as a read-through cache.  Identical to the pre-Gap-5 path.
    fn insert_on_insertion(&self, batch: RecordBatch) -> Result<(), ArrowError> {
        let bytes = serialize_batch(&batch)?;
        let length = bytes.len();
        let checksum = xxh3_64(&bytes);
        self.check_disk_capacity(length)?;
        let offset = self.write_ipc_to_disk(&bytes, checksum)?;

        let batch_arc = Arc::new(batch);
        let mut idx = self.index.write().map_err(poisoned_lock_err)?;
        let batch_idx = idx.entries.len() as u64;
        idx.hot.try_insert(batch_idx, batch_arc, length);
        idx.entries.push(Some(DiskEntry {
            file_offset: offset,
            ipc_len: length,
            checksum,
        }));
        Ok(())
    }

    /// `OnEviction`: keep the batch in the hot layer; only the batches evicted to
    /// make room are written to disk.  Two-phase commit keeps evicted batches
    /// readable from hot throughout the disk write.
    fn insert_on_eviction(&self, batch: RecordBatch) -> Result<(), ArrowError> {
        // Hot-budget accounting note: under OnEviction a batch enters the hot layer
        // *before* it is ever serialised, so its true IPC byte length is unknown
        // here.  We use the in-memory size (`get_array_memory_size`) as a proxy for
        // the budget.  This is deliberately an estimate: `hot.used` may drift from
        // the actual on-disk size, since Arrow's in-memory layout (padding, child
        // arrays, dictionaries) rarely equals the IPC-serialised size.  The drift
        // only affects *when* eviction triggers, never correctness — every batch is
        // still serialised with its exact length at eviction time.  Storing the
        // serialised bytes in hot to get an exact figure would double memory use for
        // pending batches; not worth it unless drift proves a problem in practice.
        let mem_size: usize = batch
            .columns()
            .iter()
            .map(|c| c.get_array_memory_size())
            .sum();
        let batch_arc = Arc::new(batch);

        // Phase 1: stage evictions and append the placeholder slot, under the lock.
        let evicted = {
            let mut idx = self.index.write().map_err(poisoned_lock_err)?;
            let batch_idx = idx.entries.len() as u64;
            let evicted = idx.hot.stage_evictions(batch_idx, batch_arc, mem_size);
            idx.entries.push(None); // "in hot, not yet on disk"
            evicted
        }; // lock released — evicted batches remain in hot for readers

        // Phase 2: write each evicted batch to disk (no lock held).
        let mut disk_entries: Vec<(u64, DiskEntry)> = Vec::with_capacity(evicted.len());
        for (evicted_idx, evicted_arc, _est) in &evicted {
            let bytes = serialize_batch(evicted_arc)?;
            let length = bytes.len();
            let checksum = xxh3_64(&bytes);
            self.check_disk_capacity(length)?;
            let offset = self.write_ipc_to_disk(&bytes, checksum)?;
            disk_entries.push((
                *evicted_idx,
                DiskEntry {
                    file_offset: offset,
                    ipc_len: length,
                    checksum,
                },
            ));
        }

        // Phase 3: pop the evicted entries from hot and record their disk slots,
        // atomically under one write lock so readers never see a gap.
        let mut idx = self.index.write().map_err(poisoned_lock_err)?;
        idx.hot.commit_evictions(evicted.len());
        for (evicted_idx, disk_entry) in disk_entries {
            idx.entries[evicted_idx as usize] = Some(disk_entry);
        }
        Ok(())
    }
}

enum CacheTier {
    Memory(MemoryCacheTier),
    Disk(DiskCacheTier),
}

impl CacheTier {
    /// Store a batch.  Called while `DatasetInner`'s mutex is held.
    fn insert(&self, batch: RecordBatch) -> Result<(), ArrowError> {
        let batch_size: usize = batch
            .columns()
            .iter()
            .map(|c| c.get_array_memory_size())
            .sum();
        if batch_size > MAX_BATCH_BYTES {
            return Err(other_arrow_err(format!(
                "Single batch ({batch_size} bytes) exceeds the {} byte Arrow limit",
                MAX_BATCH_BYTES
            )));
        }
        match self {
            CacheTier::Memory(m) => {
                let used = m.used.load(Ordering::Relaxed);
                if used + batch_size > m.capacity {
                    return Err(ArrowError::MemoryError(format!(
                        "Memory cache capacity ({} bytes) exceeded",
                        m.capacity
                    )));
                }
                m.used.fetch_add(batch_size, Ordering::Relaxed);
                m.batches.write().map_err(poisoned_lock_err)?.push(Arc::new(batch));
                Ok(())
            }
            CacheTier::Disk(d) => match d.policy {
                WritePolicy::OnInsertion => d.insert_on_insertion(batch),
                WritePolicy::OnEviction => d.insert_on_eviction(batch),
            },
        }
    }

    /// Retrieve batch `idx`, or `None` if not yet ingested.
    fn get(&self, idx: u64) -> Result<Option<Arc<RecordBatch>>, ArrowError> {
        match self {
            CacheTier::Memory(m) => Ok(m.batches.read().map_err(poisoned_lock_err)?.get(idx as usize).cloned()),
            CacheTier::Disk(d) => {
                // Copy the index slot and check hot while holding the read lock.
                // Both are read from the SAME snapshot so an `OnEviction` batch is
                // never observed as neither hot nor on-disk (the commit that pops it
                // from hot and the write that sets its slot happen under one lock).
                let (slot, maybe_hot) = {
                    let index = d.index.read().map_err(poisoned_lock_err)?;
                    match index.entries.get(idx as usize) {
                        None => return Ok(None),
                        Some(&slot) => (slot, index.hot.get(idx)),
                    }
                }; // read lock released before any I/O

                if let Some(arc) = maybe_hot {
                    return Ok(Some(arc)); // hot hit: no disk I/O, no checksum needed
                }

                // Not hot: the slot must be on disk by now.  A `None` here would
                // mean an evicted batch vanished from both tiers — a bug, not user data.
                let entry = slot.ok_or_else(|| {
                    other_arrow_err(format!(
                        "Batch {idx} is neither in the hot layer nor on disk \
                         — cache index inconsistency"
                    ))
                })?;

                // Disk read: pull the 8-byte header and the IPC payload in one
                // pread.  The header is the on-disk source of truth; the RAM
                // checksum guards against header corruption itself.
                let mut buf = vec![0u8; 8 + entry.ipc_len];
                pread_exact(&d.read_file, &mut buf, entry.file_offset).map_err(|e| {
                    // A short read here means the cache file was truncated/corrupted;
                    // surface it as an I/O error (→ PyIOError) like a checksum mismatch.
                    ArrowError::IoError(format!("Disk read failed: {e}"), e)
                })?;

                let header =
                    u64::from_le_bytes(buf[..8].try_into().expect("slice is exactly 8 bytes"));
                let payload = &buf[8..];
                let actual = xxh3_64(payload);
                if actual != entry.checksum || header != entry.checksum {
                    return Err(ArrowError::IoError(
                        format!(
                            "Checksum mismatch for batch {idx}: \
                             expected {:#018x}, header {:#018x}, payload {:#018x} \
                             — cache file may be corrupted",
                            entry.checksum, header, actual
                        ),
                        std::io::Error::other("checksum mismatch"),
                    ));
                }

                deserialize_batch(payload).map(|b| Some(Arc::new(b)))
            }
        }
    }

    /// Release in-memory data (hot layer and, for memory tiers, all batches).
    /// Silently ignores poisoned locks since this runs during teardown.
    fn clear(&self) {
        match self {
            CacheTier::Memory(m) => {
                if let Ok(mut batches) = m.batches.write() {
                    batches.clear();
                }
                m.used.store(0, Ordering::Relaxed);
            }
            CacheTier::Disk(d) => {
                if let Ok(mut index) = d.index.write() {
                    index.hot.clear();
                }
            }
        }
    }

    /// Delete the on-disk subdirectory (no-op for memory tiers).
    fn cleanup_disk(&self) {
        if let CacheTier::Disk(d) = self {
            let _ = std::fs::remove_dir_all(&d.dir_path);
        }
    }
}

// ── platform-portable positional read ────────────────────────────────────────

/// Read exactly `buf.len()` bytes from `file` starting at `offset` without
/// modifying the file's seek position.  Concurrent callers are safe because
/// neither Unix `pread(2)` nor Windows `ReadFile` with OVERLAPPED updates the
/// file descriptor's offset.
fn pread_exact(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut remaining = buf;
        let mut off = offset;
        while !remaining.is_empty() {
            let n = file.seek_read(remaining, off)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected EOF reading cache file",
                ));
            }
            off += n as u64;
            remaining = &mut remaining[n..];
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positional reads not supported on this platform",
        ))
    }
}

// ── error helpers ────────────────────────────────────────────────────────────

#[cold]
fn other_arrow_err(msg: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(Box::new(std::io::Error::other(msg.to_string())))
}

fn poisoned_lock_err<T>(_: std::sync::PoisonError<T>) -> ArrowError {
    other_arrow_err("Internal lock poisoned by a prior panic")
}

fn poisoned_lock_pyerr<T>(_: std::sync::PoisonError<T>) -> PyErr {
    PyRuntimeError::new_err("Internal lock poisoned by a prior panic")
}

/// Semantic error type for `without_gil` closures, so each kind maps to the
/// right Python exception at the boundary rather than everything becoming OSError.
enum BoundaryError {
    Value(String),
    Io(String),
    Memory(String),
    Runtime(String),
    Arrow(String),
}

impl From<BoundaryError> for PyErr {
    fn from(e: BoundaryError) -> PyErr {
        match e {
            BoundaryError::Value(s) => PyValueError::new_err(s),
            BoundaryError::Io(s) => PyIOError::new_err(s),
            BoundaryError::Memory(s) => PyMemoryError::new_err(s),
            BoundaryError::Runtime(s) => PyRuntimeError::new_err(s),
            BoundaryError::Arrow(s) => arrow_pyarrow::PyArrowException::new_err(s),
        }
    }
}

/// Classify an [`ArrowError`] into the right [`BoundaryError`] variant.
fn arrow_to_boundary(e: ArrowError) -> BoundaryError {
    match e {
        ArrowError::MemoryError(msg) => BoundaryError::Memory(msg),
        ArrowError::InvalidArgumentError(msg) => BoundaryError::Value(msg),
        ArrowError::IoError(msg, _) => BoundaryError::Io(msg),
        _ => BoundaryError::Arrow(e.to_string()),
    }
}

/// Convert an [`ArrowError`] into the appropriate Python exception.
fn arrow_to_py_err(e: ArrowError) -> PyErr {
    match e {
        ArrowError::MemoryError(msg) => PyMemoryError::new_err(msg),
        ArrowError::InvalidArgumentError(msg) => PyValueError::new_err(msg),
        _ => arrow_pyarrow::PyArrowException::new_err(e.to_string()),
    }
}

/// Export `schema` as an `"arrow_schema"` [`PyCapsule`] via Arrow FFI —
/// no pyarrow Python object is allocated.
fn to_schema_pycapsule<'py>(py: Python<'py>, schema: &Schema) -> PyResult<Bound<'py, PyCapsule>> {
    let ffi_schema =
        FFI_ArrowSchema::try_from(schema).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let name = CString::new("arrow_schema").map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyCapsule::new(py, ffi_schema, Some(name))
}

// ── IPC serialization ────────────────────────────────────────────────────────

fn serialize_batch(batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
    let mut buf = Vec::new();
    {
        let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

// Arrow's practical per-batch limit (int32 offset arrays cap at ~2^31 elements).
const MAX_BATCH_BYTES: usize = 2 * 1024 * 1024 * 1024;

fn deserialize_batch(bytes: &[u8]) -> Result<RecordBatch, ArrowError> {
    if bytes.len() > MAX_BATCH_BYTES {
        return Err(other_arrow_err(format!(
            "Batch size {} exceeds maximum of {} bytes",
            bytes.len(),
            MAX_BATCH_BYTES
        )));
    }
    let mut reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)?;
    reader
        .next()
        .ok_or_else(|| other_arrow_err("Empty IPC stream in cache entry"))?
}

// ── GIL management ───────────────────────────────────────────────────────────
//
// We use raw CPython FFI instead of PyO3's safe wrappers because:
//  - `py.allow_threads(f)` requires `F: Ungil`, which forbids reacquiring the
//    GIL inside `f`.  `ingest_up_to` needs to reacquire for `upstream.next()`.
//  - `Python::with_gil()` was removed in PyO3 0.28.
//
// Safety contract: `without_gil` saves the thread state and restores it on
// drop; `with_gil_acquired` ensures/releases via `PyGILState_Ensure/Release`.
// Both are no-ops under the free-threaded build (`Py_GIL_DISABLED`).

fn without_gil<T, F: FnOnce() -> T>(_py: Python<'_>, f: F) -> T {
    struct RestoreGuard(*mut pyo3::ffi::PyThreadState);
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            unsafe { pyo3::ffi::PyEval_RestoreThread(self.0) };
        }
    }
    #[cfg(not(Py_GIL_DISABLED))]
    let _guard = RestoreGuard(unsafe { pyo3::ffi::PyEval_SaveThread() });
    f()
}

fn with_gil_acquired<T, F: FnOnce() -> T>(f: F) -> T {
    #[cfg(not(Py_GIL_DISABLED))]
    {
        struct ReleaseGuard(pyo3::ffi::PyGILState_STATE);
        impl Drop for ReleaseGuard {
            fn drop(&mut self) {
                unsafe { pyo3::ffi::PyGILState_Release(self.0) };
            }
        }
        let _guard = ReleaseGuard(unsafe { pyo3::ffi::PyGILState_Ensure() });
        f()
    }
    #[cfg(Py_GIL_DISABLED)]
    f()
}

// ── shared ingestion state ───────────────────────────────────────────────────

struct DatasetInner {
    cache: Arc<CacheTier>,
    upstream: Option<Box<dyn arrow_array::RecordBatchReader + Send>>,
    ingested_count: u64,
    upstream_exhausted: bool,
    closed: bool,
}

impl DatasetInner {
    fn ingest_up_to(&mut self, target_index: u64) -> Result<bool, ArrowError> {
        if self.closed {
            return Err(ArrowError::InvalidArgumentError(
                "Dataset has been closed".into(),
            ));
        }
        while self.ingested_count <= target_index {
            if self.upstream_exhausted {
                return Ok(false);
            }
            let next_result = with_gil_acquired(|| self.upstream.as_mut().and_then(|u| u.next()));
            let batch = match next_result {
                None => {
                    self.upstream_exhausted = true;
                    return Ok(false);
                }
                Some(Err(e)) => return Err(e),
                Some(Ok(b)) => b,
            };
            self.cache.insert(batch)?;
            self.ingested_count += 1;
        }
        Ok(true)
    }
}

// ── StreamCacheReaderImpl ──────────────────────────────────────────────────

struct StreamCacheReaderImpl {
    schema: SchemaRef,
    inner: Arc<Mutex<DatasetInner>>,
    current_index: u64,
}

impl Iterator for StreamCacheReaderImpl {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;

        // Ensure the batch has been ingested and clone the cache handle.
        let cache = {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(e) => return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}")))),
            };
            if inner.closed {
                return Some(Err(ArrowError::InvalidArgumentError(
                    "Dataset has been closed".into(),
                )));
            }
            match inner.ingest_up_to(idx) {
                Err(e) => return Some(Err(e)),
                Ok(false) => return None,
                Ok(true) => {}
            }
            inner.cache.clone()
        }; // mutex released before I/O

        // Fetch from cache; no lock held — safe for concurrent readers.
        match cache.get(idx) {
            Err(e) => Some(Err(e)),
            Ok(None) => {
                let closed = self.inner.lock().map(|g| g.closed).unwrap_or(false);
                if closed {
                    Some(Err(ArrowError::InvalidArgumentError(
                        "Dataset has been closed".into(),
                    )))
                } else {
                    // Should not happen: ingest_up_to returned Ok(true).
                    Some(Err(other_arrow_err(format!(
                        "Batch {idx} disappeared from the cache unexpectedly"
                    ))))
                }
            }
            Ok(Some(arc)) => {
                self.current_index += 1;
                // Clone the RecordBatch (cheap: clones Arc pointers to buffers).
                Some(Ok((*arc).clone()))
            }
        }
    }
}

impl arrow_array::RecordBatchReader for StreamCacheReaderImpl {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ── PyStreamCacheReader ────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCacheReader", frozen)]
pub struct PyStreamCacheReader(Mutex<Option<StreamCacheReaderImpl>>);

impl PyStreamCacheReader {
    fn new(impl_: StreamCacheReaderImpl) -> Self {
        Self(Mutex::new(Some(impl_)))
    }

    fn to_stream_pycapsule<'py>(
        py: Python<'py>,
        reader: StreamCacheReaderImpl,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let boxed: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(reader);
        match requested_schema {
            None => {
                // Fast path: export directly via Arrow C Stream FFI, no Python allocation.
                let ffi_stream = FFI_ArrowArrayStream::new(boxed);
                let name = CString::new("arrow_array_stream")
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                PyCapsule::new(py, ffi_stream, Some(name))
            }
            Some(schema) => {
                // Schema-casting path: let pyarrow handle the requested_schema negotiation.
                let py_reader = boxed.into_pyarrow(py)?;
                py_reader
                    .call_method1("__arrow_c_stream__", (schema,))?
                    .cast_into::<PyCapsule>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "pyarrow __arrow_c_stream__ did not return a PyCapsule",
                        )
                    })
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCacheReader {
    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self
            .0
            .lock()
            .map_err(poisoned_lock_pyerr)?
            .take()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Self::to_stream_pycapsule(py, reader, requested_schema.map(|c| c.into_any()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().map_err(poisoned_lock_pyerr)?;
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        to_schema_pycapsule(py, reader.schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        let inner = self.0.lock().map_err(poisoned_lock_pyerr)?;
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Ok(PyArrowType((*reader.schema).clone()))
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self.0.lock().map_err(poisoned_lock_pyerr)?.is_none())
    }

    fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn cast<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let impl_ = self
            .0
            .lock()
            .map_err(poisoned_lock_pyerr)?
            .take()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        let new_reader = PyStreamCacheReader::new(impl_);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("schema", target_schema)?;
        let py_reader = Bound::new(py, new_reader)?;
        py.import("pyarrow")?
            .getattr("RecordBatchReader")?
            .getattr("from_stream")?
            .call((py_reader,), Some(&kwargs))
    }

    #[gen_stub(override_return_type(type_repr = "pa.RecordBatch", imports = ("pyarrow as pa",)))]
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyArrowType<RecordBatch>>> {
        let mut guard = self.0.lock().map_err(poisoned_lock_pyerr)?;
        let impl_ = match guard.as_mut() {
            None => {
                return Err(PyValueError::new_err("Reader already consumed"));
            }
            Some(r) => r,
        };
        let result = without_gil(py, || impl_.next());
        match result {
            None => Ok(None),
            Some(Err(e)) => Err(arrow_to_py_err(e)),
            Some(Ok(batch)) => Ok(Some(PyArrowType(batch))),
        }
    }
}

// ── PyCastingStreamCache ──────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "CastingStreamCache", frozen)]
pub struct PyCastingStreamCache {
    inner: Arc<Mutex<DatasetInner>>,
    source_schema: SchemaRef,
    target_schema: SchemaRef,
}

impl PyCastingStreamCache {
    fn make_reader_impl(&self, py: Python<'_>) -> PyResult<StreamCacheReaderImpl> {
        without_gil(py, || {
            let inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            if inner.closed {
                return Err(BoundaryError::Value("Dataset has been closed".into()));
            }
            Ok::<_, BoundaryError>(StreamCacheReaderImpl {
                schema: self.source_schema.clone(),
                inner: self.inner.clone(),
                current_index: 0,
            })
        })
        .map_err(PyErr::from)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCastingStreamCache {
    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        Ok(PyArrowType((*self.target_schema).clone()))
    }

    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let impl_ = self.make_reader_impl(py)?;
        let effective_schema: Option<Bound<'py, PyAny>> = if requested_schema.is_some() {
            requested_schema.map(|c| c.into_any())
        } else {
            Some(to_schema_pycapsule(py, self.target_schema.as_ref())?.into_any())
        };
        PyStreamCacheReader::to_stream_pycapsule(py, impl_, effective_schema)
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.target_schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "CastingStreamCache", imports = ()))]
    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PyArrowType<Schema>,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            inner: self.inner.clone(),
            source_schema: self.source_schema.clone(),
            target_schema: Arc::new(target_schema.0),
        })
    }
}

// ── PyStreamCache ─────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCache", frozen)]
pub struct PyStreamCache {
    schema: SchemaRef,
    inner: Arc<Mutex<DatasetInner>>,
}

impl Drop for PyStreamCache {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock()
            && !inner.closed
        {
            inner.closed = true;
            let cache = inner.cache.clone();
            drop(inner);
            cache.cleanup_disk();
            cache.clear();
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCache {
    #[new]
    #[pyo3(signature = (reader, memory_capacity = None, disk_path = None, disk_capacity = None, write_policy = "on_insertion".to_string()))]
    pub fn new(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        reader: PyArrowType<ArrowArrayStreamReader>,
        memory_capacity: Option<usize>,
        disk_path: Option<String>,
        disk_capacity: Option<u64>,
        write_policy: String,
    ) -> PyResult<Self> {
        let upstream: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(reader.0);
        let schema = upstream.schema();

        // Parse the write policy up front (applies to the disk tier only; ignored
        // for memory-only caches).
        let policy = match write_policy.as_str() {
            "on_insertion" => WritePolicy::OnInsertion,
            "on_eviction" => WritePolicy::OnEviction,
            other => {
                return Err(PyValueError::new_err(format!(
                    "write_policy must be 'on_insertion' or 'on_eviction', got {other:?}"
                )));
            }
        };

        let cache = match (disk_path, disk_capacity) {
            (Some(path), Some(capacity)) => {
                let id = DATASET_COUNTER.fetch_add(1, Ordering::Relaxed);
                let dir_path = PathBuf::from(&path).join(format!("_{id}"));

                // Create the subdirectory and both file descriptors while the
                // GIL is released (pure OS I/O — no Python objects touched).
                without_gil(py, || {
                    std::fs::create_dir_all(&dir_path).map_err(|e| {
                        BoundaryError::Io(format!("Failed to create cache directory: {e}"))
                    })?;
                    let file_path = dir_path.join("cache.arrow");
                    let mut open_opts = std::fs::OpenOptions::new();
                    open_opts.write(true).create_new(true);
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::OpenOptionsExt;
                        open_opts.mode(0o600);
                    }
                    let write_file = open_opts.open(&file_path).map_err(|e| {
                        BoundaryError::Io(format!("Failed to create cache file: {e}"))
                    })?;
                    let read_file = std::fs::File::open(&file_path).map_err(|e| {
                        BoundaryError::Io(format!("Failed to open cache file for reading: {e}"))
                    })?;

                    let hot_capacity = memory_capacity.unwrap_or_else(|| {
                        // Cap at half of system RAM so multiple caches don't
                        // collectively exhaust memory; floor at 64 MiB.
                        (*TOTAL_SYSTEM_MEMORY / 2).max(64 * 1024 * 1024)
                    });
                    Ok::<_, BoundaryError>(CacheTier::Disk(DiskCacheTier {
                        dir_path,
                        write_state: Mutex::new(DiskWriteState {
                            file: write_file,
                            offset: 0,
                        }),
                        index: RwLock::new(DiskIndex {
                            entries: Vec::new(),
                            hot: HotLayer::new(hot_capacity),
                        }),
                        read_file,
                        disk_capacity: capacity,
                        disk_used: AtomicU64::new(0),
                        policy,
                    }))
                })
                .map_err(PyErr::from)?
            }
            (None, None) => {
                let capacity = memory_capacity.unwrap_or_else(|| {
                    // Default to 10% of system RAM, floor at 64 MiB.
                    (*TOTAL_SYSTEM_MEMORY / 10).max(64 * 1024 * 1024)
                });
                CacheTier::Memory(MemoryCacheTier {
                    batches: RwLock::new(Vec::new()),
                    capacity,
                    used: AtomicUsize::new(0),
                })
            }
            _ => {
                return Err(PyValueError::new_err(
                    "disk_path and disk_capacity must both be provided, or both omitted",
                ));
            }
        };

        let inner = DatasetInner {
            cache: Arc::new(cache),
            upstream: Some(upstream),
            ingested_count: 0,
            upstream_exhausted: false,
            closed: false,
        };

        Ok(Self {
            schema,
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        Ok(PyArrowType((*self.schema).clone()))
    }

    #[pyo3(signature = (from_start = true))]
    pub fn reader(&self, py: Python<'_>, from_start: bool) -> PyResult<PyStreamCacheReader> {
        without_gil(py, || {
            let inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            if inner.closed {
                return Err(BoundaryError::Value("Dataset has been closed".into()));
            }
            let start_index = if from_start { 0 } else { inner.ingested_count };
            Ok::<_, BoundaryError>(PyStreamCacheReader::new(StreamCacheReaderImpl {
                schema: self.schema.clone(),
                inner: self.inner.clone(),
                current_index: start_index,
            }))
        })
        .map_err(PyErr::from)
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyStreamCacheReader> {
        self.reader(py, true)
    }

    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self.reader(py, true)?;
        let impl_ = reader
            .0
            .lock()
            .map_err(poisoned_lock_pyerr)?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("freshly created reader was unexpectedly consumed"))?;
        PyStreamCacheReader::to_stream_pycapsule(py, impl_, requested_schema.map(|c| c.into_any()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.schema.as_ref())
    }

    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PyArrowType<Schema>,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            inner: self.inner.clone(),
            source_schema: self.schema.clone(),
            target_schema: Arc::new(target_schema.0),
        })
    }

    pub fn ingest_all(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            inner.ingest_up_to(u64::MAX).map_err(arrow_to_boundary)?;
            Ok::<_, BoundaryError>(inner.ingested_count)
        })
        .map_err(PyErr::from)
    }

    #[getter]
    pub fn ingested_count(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || {
            self.inner
                .lock()
                .map(|g| g.ingested_count)
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))
        })
        .map_err(PyErr::from)
    }

    #[getter]
    pub fn upstream_exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        without_gil(py, || {
            self.inner
                .lock()
                .map(|g| g.upstream_exhausted)
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))
        })
        .map_err(PyErr::from)
    }

    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        without_gil(py, || {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            if inner.closed {
                return Ok(());
            }
            inner.closed = true;
            let cache = inner.cache.clone();
            drop(inner);
            cache.cleanup_disk();
            cache.clear();
            Ok::<_, BoundaryError>(())
        })
        .map_err(PyErr::from)
    }
}

// ── unit tests ─────────────────────────────────────────────────────────────────
//
// These exercise the pure cache machinery (HotLayer FIFO, the disk index state
// machine, and CacheTier::get) directly in Rust.  The Python suite covers the
// end-to-end behaviour; these cover the defensive branches that are unreachable
// through the public API but must still behave correctly.

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field};

    fn make_batch(vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let arr = Arc::new(Int32Array::from(vals.to_vec()));
        RecordBatch::try_new(schema, vec![arr]).expect("valid batch")
    }

    fn batch_mem_size(batch: &RecordBatch) -> usize {
        batch
            .columns()
            .iter()
            .map(|c| c.get_array_memory_size())
            .sum()
    }

    /// Build a disk tier backed by a fresh temp file.  Uses the global dataset
    /// counter to avoid name collisions across parallel tests.
    fn temp_disk_tier(
        disk_capacity: u64,
        hot_capacity: usize,
        policy: WritePolicy,
    ) -> DiskCacheTier {
        let id = DATASET_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::env::temp_dir().join(format!("batchcorder_unit_{id}"));
        std::fs::create_dir_all(&dir_path).expect("create temp dir");
        let file_path = dir_path.join("cache.arrow");
        let write_file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .expect("open write file");
        let read_file = std::fs::File::open(&file_path).expect("open read file");
        DiskCacheTier {
            dir_path,
            write_state: Mutex::new(DiskWriteState {
                file: write_file,
                offset: 0,
            }),
            index: RwLock::new(DiskIndex {
                entries: Vec::new(),
                hot: HotLayer::new(hot_capacity),
            }),
            read_file,
            disk_capacity,
            disk_used: AtomicU64::new(0),
            policy,
        }
    }

    // ── HotLayer ──────────────────────────────────────────────────────────────

    #[test]
    fn hot_layer_insert_and_get() {
        let mut hot = HotLayer::new(1024);
        let b = Arc::new(make_batch(&[1, 2, 3]));
        assert!(hot.try_insert(0, b.clone(), 16));
        assert!(hot.get(0).is_some());
        // Index below head and beyond the tail both miss.
        assert!(hot.get(1).is_none());
    }

    #[test]
    fn hot_layer_get_below_head_returns_none() {
        let mut hot = HotLayer::new(32);
        // Two small entries, then force the first out so head advances past 0.
        hot.try_insert(0, Arc::new(make_batch(&[0])), 16);
        hot.try_insert(1, Arc::new(make_batch(&[1])), 16);
        hot.try_insert(2, Arc::new(make_batch(&[2])), 16); // evicts batch 0
        assert!(hot.get(0).is_none()); // below head
        assert!(hot.get(2).is_some());
    }

    #[test]
    fn hot_layer_oversized_not_stored() {
        let mut hot = HotLayer::new(8);
        // Larger than the whole budget: try_insert returns false and stores nothing.
        assert!(!hot.try_insert(0, Arc::new(make_batch(&[0])), 64));
        assert!(hot.get(0).is_none());
    }

    #[test]
    fn hot_layer_oversized_then_fitting_keeps_index_mapping() {
        // Regression: with variable batch sizes an oversized batch is rejected
        // (not cached), but a later fitting batch must still be readable under
        // its own index.  The rejected batch has to advance head so the mapping
        // `entries[0] == head_batch_idx` holds.
        let mut hot = HotLayer::new(32);
        assert!(!hot.try_insert(0, Arc::new(make_batch(&[0])), 64)); // oversized
        assert!(!hot.try_insert(1, Arc::new(make_batch(&[1])), 64)); // oversized
        assert!(hot.try_insert(2, Arc::new(make_batch(&[2])), 16)); // fits
        assert_eq!(hot.head_batch_idx, 2);
        assert!(hot.get(0).is_none());
        assert!(hot.get(1).is_none());
        // Must return batch 2's data, not a misindexed slot.
        let got = hot.get(2).expect("batch 2 readable");
        assert_eq!(got.as_ref(), &make_batch(&[2]));
    }

    #[test]
    fn hot_layer_fifo_eviction_order() {
        let mut hot = HotLayer::new(40); // ~2 entries of 16 bytes
        for i in 0..4u64 {
            hot.try_insert(i, Arc::new(make_batch(&[i as i32])), 16);
        }
        // Oldest evicted first; only the two most recent remain.
        assert!(hot.get(0).is_none());
        assert!(hot.get(1).is_none());
        assert!(hot.get(2).is_some());
        assert!(hot.get(3).is_some());
    }

    #[test]
    fn hot_layer_stage_and_commit_evictions() {
        let mut hot = HotLayer::new(40);
        hot.try_insert(0, Arc::new(make_batch(&[0])), 16);
        hot.try_insert(1, Arc::new(make_batch(&[1])), 16);
        // Staging a third entry returns batch 0 as the eviction candidate but does
        // NOT remove it yet — it stays readable.
        let evicted = hot.stage_evictions(2, Arc::new(make_batch(&[2])), 16);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, 0);
        assert!(
            hot.get(0).is_some(),
            "staged-but-not-committed stays readable"
        );
        // Commit removes it.
        hot.commit_evictions(evicted.len());
        assert!(hot.get(0).is_none());
        assert!(hot.get(2).is_some());
    }

    #[test]
    fn hot_layer_commit_more_than_present_is_safe() {
        // Exercises the `None => break` arm of commit_evictions: asking to commit
        // more entries than exist must stop at empty instead of underflowing.
        let mut hot = HotLayer::new(1024);
        hot.try_insert(0, Arc::new(make_batch(&[0])), 16);
        hot.commit_evictions(5); // only 1 present
        assert_eq!(hot.used, 0);
        assert!(hot.get(0).is_none());
    }

    #[test]
    fn hot_layer_stage_with_no_room_to_free() {
        // Capacity smaller than the newcomer and nothing evictable: stage_evictions
        // breaks out of its loop and still appends the newcomer.
        let mut hot = HotLayer::new(8);
        let evicted = hot.stage_evictions(0, Arc::new(make_batch(&[0])), 64);
        assert!(evicted.is_empty());
        assert!(hot.get(0).is_some());
    }

    // ── CacheTier::get (disk) defensive branches ────────────────────────────────

    #[test]
    fn disk_get_out_of_range_returns_none() {
        // Line: `None => return Ok(None)` — index past the end of `entries`.
        let tier = CacheTier::Disk(temp_disk_tier(1 << 20, 1 << 20, WritePolicy::OnInsertion));
        assert!(tier.get(99).expect("ok").is_none());
        tier.cleanup_disk();
    }

    #[test]
    fn disk_get_none_slot_not_hot_is_inconsistency_error() {
        // Lines: `slot.ok_or_else(...)` — an entry that is neither on disk (None
        // slot) nor in the hot layer is a cache-index inconsistency, not user data.
        let d = temp_disk_tier(1 << 20, 1 << 20, WritePolicy::OnInsertion);
        d.index.write().unwrap().entries.push(None); // placeholder, never committed
        let tier = CacheTier::Disk(d);
        let err = tier.get(0).expect_err("must be an error");
        assert!(err.to_string().contains("inconsistency"));
        tier.cleanup_disk();
    }

    // ── insert + read round-trips ───────────────────────────────────────────────

    #[test]
    fn disk_on_insertion_reads_back_from_disk() {
        // hot_capacity of 1 forces the batch out of hot, so get() must read it back
        // from disk and pass the checksum verification.
        let tier = CacheTier::Disk(temp_disk_tier(1 << 20, 1, WritePolicy::OnInsertion));
        let batch = make_batch(&[10, 20, 30]);
        tier.insert(batch.clone()).expect("insert");
        let got = tier.get(0).expect("ok").expect("present");
        assert_eq!(got.as_ref(), &batch);
        tier.cleanup_disk();
    }

    #[test]
    fn disk_on_eviction_writes_only_evicted_batches() {
        // Tiny hot: each new batch evicts the previous one to disk.  The most recent
        // batch is served from hot; older ones from disk.
        let first = make_batch(&[1]);
        let mem = batch_mem_size(&first);
        let tier = CacheTier::Disk(temp_disk_tier(1 << 20, mem, WritePolicy::OnEviction));
        tier.insert(first.clone()).expect("insert 0");
        tier.insert(make_batch(&[2])).expect("insert 1");
        tier.insert(make_batch(&[3])).expect("insert 2");

        // Batch 0 was evicted to disk and reads back correctly.
        assert_eq!(tier.get(0).expect("ok").expect("present").as_ref(), &first);
        // Batch 2 (most recent) is still hot.
        assert_eq!(
            tier.get(2).expect("ok").expect("present").as_ref(),
            &make_batch(&[3])
        );
        tier.cleanup_disk();
    }

    #[test]
    fn disk_capacity_exceeded_errors() {
        // disk_capacity smaller than 8 + ipc_len: the capacity check fires.
        let tier = CacheTier::Disk(temp_disk_tier(4, 1, WritePolicy::OnInsertion));
        let err = tier.insert(make_batch(&[1, 2, 3])).expect_err("capacity");
        assert!(matches!(err, ArrowError::MemoryError(_)));
        tier.cleanup_disk();
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let batch = make_batch(&[7, 8, 9]);
        let bytes = serialize_batch(&batch).expect("serialize");
        let back = deserialize_batch(&bytes).expect("deserialize");
        assert_eq!(back, batch);
    }

    // ── Memory tier ─────────────────────────────────────────────────────────────

    fn memory_tier(capacity: usize) -> CacheTier {
        CacheTier::Memory(MemoryCacheTier {
            batches: RwLock::new(Vec::new()),
            capacity,
            used: AtomicUsize::new(0),
        })
    }

    #[test]
    fn memory_tier_insert_get_and_clear() {
        let tier = memory_tier(1 << 20);
        tier.insert(make_batch(&[1])).expect("insert 0");
        tier.insert(make_batch(&[2])).expect("insert 1");
        assert!(tier.get(0).expect("ok").is_some());
        assert!(tier.get(1).expect("ok").is_some());
        assert!(tier.get(2).expect("ok").is_none()); // beyond end
        tier.clear();
        assert!(tier.get(0).expect("ok").is_none());
    }

    #[test]
    fn memory_tier_capacity_exceeded_errors() {
        let tier = memory_tier(4); // smaller than one batch
        let err = tier.insert(make_batch(&[1, 2, 3])).expect_err("capacity");
        assert!(matches!(err, ArrowError::MemoryError(_)));
    }

    // ── disk corruption detection (mirrors the Python error tests) ───────────────

    #[test]
    fn disk_checksum_mismatch_is_detected() {
        let d = temp_disk_tier(1 << 20, 1, WritePolicy::OnInsertion); // hot=1 → disk read
        let path = d.dir_path.join("cache.arrow");
        let tier = CacheTier::Disk(d);
        tier.insert(make_batch(&[1, 2, 3])).expect("insert");

        // Flip a byte in the middle of the payload (offset 8 is the IPC
        // continuation marker 0xFFFFFFFF, so writing 0xFF there is a no-op — pick a
        // mid-payload byte and XOR it to guarantee a real change).
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("reopen");
        let len = f.metadata().expect("metadata").len();
        let mid = 8 + (len - 8) / 2;
        f.seek(SeekFrom::Start(mid)).expect("seek");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read byte");
        f.seek(SeekFrom::Start(mid)).expect("seek back");
        f.write_all(&[byte[0] ^ 0xFF]).expect("corrupt");
        f.flush().expect("flush");

        let err = tier.get(0).expect_err("must detect corruption");
        assert!(err.to_string().contains("Checksum mismatch"));
        tier.cleanup_disk();
    }

    #[test]
    fn disk_clear_empties_hot_but_keeps_disk_readable() {
        // CacheTier::clear on a disk tier drops the hot layer (HotLayer::clear) but
        // leaves the on-disk data intact, so reads still succeed afterwards.
        let tier = CacheTier::Disk(temp_disk_tier(1 << 20, 1 << 20, WritePolicy::OnInsertion));
        let batch = make_batch(&[5, 6]);
        tier.insert(batch.clone()).expect("insert");
        tier.clear(); // empties the hot layer
        // Still readable — served from disk now that hot is empty.
        assert_eq!(tier.get(0).expect("ok").expect("present").as_ref(), &batch);
        tier.cleanup_disk();
    }

    #[test]
    fn disk_truncated_read_errors() {
        let d = temp_disk_tier(1 << 20, 1, WritePolicy::OnInsertion);
        let path = d.dir_path.join("cache.arrow");
        let tier = CacheTier::Disk(d);
        tier.insert(make_batch(&[1, 2, 3])).expect("insert");

        // Chop the file so the payload can no longer be fully read.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen")
            .set_len(4)
            .expect("truncate");

        let err = tier.get(0).expect_err("must fail on short read");
        assert!(matches!(err, ArrowError::IoError(_, _)));
        tier.cleanup_disk();
    }
}
