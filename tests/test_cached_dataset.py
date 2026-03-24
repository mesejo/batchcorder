import threading

import pyarrow as pa
import pytest

from batchcorder import CachedDataset


def _make_table(n_batches: int = 4, rows_per_batch: int = 3) -> pa.Table:
    n = n_batches * rows_per_batch
    return pa.table(
        {
            "id": list(range(n)),
            "value": [float(i) * 1.5 for i in range(n)],
            "label": [f"row_{i}" for i in range(n)],
        }
    )


def _dataset(tmp_path, table=None, batch_size=3, mem_mb=16, disk_mb=64):
    if table is None:
        table = _make_table()
    return CachedDataset(
        table.to_reader(max_chunksize=batch_size),
        memory_capacity=mem_mb * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=disk_mb * 1024 * 1024,
    )


def _schema_from_capsule(capsule) -> pa.Schema:
    class _Wrap:
        def __arrow_c_schema__(self):
            return capsule

    return pa.schema(_Wrap())


def test_basic_construction(tmp_path):
    assert _dataset(tmp_path) is not None


def test_accepts_pyarrow_table_directly(tmp_path):
    ds = CachedDataset(
        _make_table(),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    assert ds is not None


def test_ingested_count_starts_at_zero(tmp_path):
    assert _dataset(tmp_path).ingested_count == 0


def test_upstream_not_exhausted_initially(tmp_path):
    assert _dataset(tmp_path).upstream_exhausted is False


def test_schema_via_c_schema_capsule(tmp_path):
    table = _make_table()
    schema = _schema_from_capsule(_dataset(tmp_path, table).__arrow_c_schema__())
    assert schema == table.schema


def test_schema_property_matches_source(tmp_path):
    table = _make_table()
    assert pa.schema(_dataset(tmp_path, table).schema) == table.schema


def test_reader_schema_matches_dataset(tmp_path):
    table = _make_table()
    assert pa.schema(_dataset(tmp_path, table).reader().schema) == table.schema


def test_reader_returns_correct_data(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    result = pa.RecordBatchReader.from_stream(
        _dataset(tmp_path, table).reader()
    ).read_all()
    assert result.equals(table)


def test_dataset_c_stream_returns_correct_data(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    result = pa.RecordBatchReader.from_stream(_dataset(tmp_path, table)).read_all()
    assert result.equals(table)


def test_single_batch(tmp_path):
    table = pa.table({"x": [1, 2, 3]})
    result = pa.RecordBatchReader.from_stream(
        _dataset(tmp_path, table, batch_size=100)
    ).read_all()
    assert result.equals(table)


def test_many_small_batches(tmp_path):
    table = _make_table(n_batches=20, rows_per_batch=1)
    result = pa.RecordBatchReader.from_stream(
        _dataset(tmp_path, table, batch_size=1)
    ).read_all()
    assert result.equals(table)


def test_empty_table(tmp_path):
    table = pa.table({"x": pa.array([], type=pa.int32())})
    result = pa.RecordBatchReader.from_stream(_dataset(tmp_path, table)).read_all()
    assert result.num_rows == 0
    assert result.schema.equals(table.schema)


def test_various_dtypes(tmp_path):
    table = pa.table(
        {
            "i32": pa.array([1, 2, 3], type=pa.int32()),
            "f64": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "utf8": pa.array(["a", "b", "c"], type=pa.string()),
            "bool": pa.array([True, False, True], type=pa.bool_()),
            "bin": pa.array([b"x", b"y", b"z"], type=pa.binary()),
        }
    )
    result = pa.RecordBatchReader.from_stream(_dataset(tmp_path, table)).read_all()
    assert result.equals(table)


def test_nullable_columns(tmp_path):
    table = pa.table(
        {
            "x": pa.array([1, None, 3], type=pa.int64()),
            "y": pa.array(["a", None, "c"], type=pa.string()),
        }
    )
    result = pa.RecordBatchReader.from_stream(
        _dataset(tmp_path, table, batch_size=2)
    ).read_all()
    assert result.equals(table)


def test_two_readers_return_same_data(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    res1 = pa.RecordBatchReader.from_stream(ds.reader()).read_all()
    res2 = pa.RecordBatchReader.from_stream(ds.reader()).read_all()
    assert res1.equals(table)
    assert res2.equals(table)


def test_five_readers_all_independent(tmp_path):
    table = _make_table(n_batches=6, rows_per_batch=2)
    ds = _dataset(tmp_path, table, batch_size=2)
    for _ in range(5):
        assert pa.RecordBatchReader.from_stream(ds.reader()).read_all().equals(table)


def test_c_stream_replays_from_start_each_call(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    assert pa.RecordBatchReader.from_stream(ds).read_all().equals(table)
    assert pa.RecordBatchReader.from_stream(ds).read_all().equals(table)


def test_dataset_iterable(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    batches = list(ds)
    assert len(batches) == 4
    assert sum(b.num_rows for b in batches) == table.num_rows


def test_reader_iterable(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    batches = list(ds.reader())
    assert len(batches) == 4
    assert sum(b.num_rows for b in batches) == table.num_rows


def test_source_reader_advances_as_dataset_ingests(tmp_path):
    # CachedDataset owns the source reader and pulls from it during ingestion.
    # Reading via a from_stream wrapper advances the source reader position.
    table = _make_table(n_batches=4, rows_per_batch=3)
    reader = table.to_reader(max_chunksize=3)
    ds = CachedDataset(
        reader,
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    wrapper = pa.RecordBatchReader.from_stream(ds)
    first = wrapper.read_next_batch()
    assert first.equals(table.slice(0, 3).to_batches()[0])
    # The source reader has been advanced past the batch the dataset consumed.
    second = reader.read_next_batch()
    assert second.equals(table.slice(3, 3).to_batches()[0])


def test_reader_closed_after_c_stream_export(tmp_path):
    ds = _dataset(tmp_path)
    r = ds.reader()
    assert r.closed is False
    pa.RecordBatchReader.from_stream(r).read_all()
    assert r.closed is True


def test_reader_second_c_stream_raises(tmp_path):
    ds = _dataset(tmp_path)
    r = ds.reader()
    pa.RecordBatchReader.from_stream(r).read_all()
    with pytest.raises(Exception, match="consumed|closed"):
        r.__arrow_c_stream__()


def test_reader_c_schema_raises_after_consumed(tmp_path):
    ds = _dataset(tmp_path)
    r = ds.reader()
    pa.RecordBatchReader.from_stream(r).read_all()
    with pytest.raises(Exception, match="consumed|closed"):
        r.__arrow_c_schema__()


def test_frontier_reader_empty_after_full_ingest(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    ds.ingest_all()
    assert (
        pa.RecordBatchReader.from_stream(ds.reader(from_start=False))
        .read_all()
        .num_rows
        == 0
    )


def test_from_start_replay_after_full_ingest(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    ds.ingest_all()
    assert (
        pa.RecordBatchReader.from_stream(ds.reader(from_start=True))
        .read_all()
        .equals(table)
    )


def test_frontier_reader_mid_stream(tmp_path):
    table = _make_table(n_batches=6, rows_per_batch=2)
    ds = _dataset(tmp_path, table, batch_size=2)
    half = pa.RecordBatchReader.from_stream(ds.reader())
    for _ in range(3):
        half.read_next_batch()
    assert ds.ingested_count == 3
    result = pa.RecordBatchReader.from_stream(ds.reader(from_start=False)).read_all()
    assert result.num_rows == 6  # remaining 3 batches × 2 rows


def test_ingest_all_returns_batch_count(tmp_path):
    assert (
        _dataset(
            tmp_path, _make_table(n_batches=5, rows_per_batch=2), batch_size=2
        ).ingest_all()
        == 5
    )


def test_ingested_count_increments_lazily(tmp_path):
    ds = _dataset(tmp_path, _make_table(n_batches=4, rows_per_batch=3))
    assert ds.ingested_count == 0
    pa.RecordBatchReader.from_stream(ds.reader()).read_next_batch()
    assert ds.ingested_count >= 1


def test_upstream_exhausted_after_ingest_all(tmp_path):
    ds = _dataset(tmp_path)
    assert ds.upstream_exhausted is False
    ds.ingest_all()
    assert ds.upstream_exhausted is True


def test_ingest_all_idempotent(tmp_path):
    table = _make_table(n_batches=4, rows_per_batch=3)
    ds = _dataset(tmp_path, table)
    assert ds.ingest_all() == ds.ingest_all() == 4


def test_disk_spill_and_pyarrow_ipc_roundtrip(tmp_path):
    # Each batch: 100 rows × 1 KiB binary payload ≈ 100 KiB serialized.
    # Memory tier is only 32 KiB — smaller than a single batch — so every
    # batch inserted into the cache must be evicted to the disk tier.
    n_batches = 5
    rows_per_batch = 100
    table = pa.table(
        {
            "id": list(range(n_batches * rows_per_batch)),
            "payload": pa.array(
                [b"x" * 1024] * (n_batches * rows_per_batch), type=pa.large_binary()
            ),
        }
    )

    ds = CachedDataset(
        table.to_reader(max_chunksize=rows_per_batch),
        memory_capacity=32 * 1024,  # 32 KiB — forces spill
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,  # 64 MiB
    )

    # Ingest all batches so the disk tier is populated.
    assert ds.ingest_all() == n_batches
    assert ds.upstream_exhausted

    # Foyer creates its block device files during cache construction and
    # populates them as batches are evicted from the memory tier.
    disk_files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(disk_files) > 0, "Expected Foyer to write cache files to disk"

    # Read back the full dataset via PyArrow's Arrow IPC stream interface.
    # __arrow_c_stream__ exposes the data as an Arrow IPC-backed C stream,
    # which pa.RecordBatchReader.from_stream() deserializes transparently.
    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.equals(table)


def test_disk_spill_ipc_file_write_and_read(tmp_path):
    # Same spill setup: 32 KiB memory tier, ~100 KiB per batch.
    n_batches = 5
    rows_per_batch = 100
    table = pa.table(
        {
            "id": list(range(n_batches * rows_per_batch)),
            "payload": pa.array(
                [b"x" * 1024] * (n_batches * rows_per_batch), type=pa.large_binary()
            ),
        }
    )

    # Separate subdirectory so the Foyer device files and the IPC output
    # file live in different places, keeping the disk-file assertion clean.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ipc_path = tmp_path / "out.arrow"

    ds = CachedDataset(
        table.to_reader(max_chunksize=rows_per_batch),
        memory_capacity=32 * 1024,  # 32 KiB — forces spill to disk
        disk_path=str(cache_dir),
        disk_capacity=64 * 1024 * 1024,
    )
    ds.ingest_all()

    # Verify Foyer wrote its block device files under cache_dir.
    cache_files = [p for p in cache_dir.rglob("*") if p.is_file()]
    assert len(cache_files) > 0, "Expected Foyer to write cache files to disk"

    # Stream the data out of the cache (reading from the disk tier) and write
    # it to a standard Arrow IPC *file* (random-access format).
    reader = pa.RecordBatchReader.from_stream(ds)
    with (
        pa.OSFile(str(ipc_path), "wb") as sink,
        pa.ipc.new_file(sink, reader.schema) as writer,
    ):
        for batch in reader:
            writer.write_batch(batch)

    assert ipc_path.exists(), "IPC file was not created"

    # Read back using a memory map (zero-copy) and verify round-trip integrity.
    # pa.memory_map + pa.ipc.open_file is the random-access, allocation-free path.
    with pa.memory_map(str(ipc_path), "rb") as source:
        loaded = pa.ipc.open_file(source).read_all()
        assert loaded.equals(table)


def test_concurrent_readers(tmp_path):
    table = _make_table(n_batches=8, rows_per_batch=5)
    ds = _dataset(tmp_path, table, batch_size=5)
    results: list[pa.Table | None] = [None] * 4
    errors: list[Exception] = []

    def read(i):
        try:
            results[i] = pa.RecordBatchReader.from_stream(ds.reader()).read_all()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for res in results:
        assert res is not None
        assert res.equals(table)


def test_close_removes_disk_files(tmp_path):
    """close() must delete the Foyer block-device files it wrote to disk."""
    n_batches = 5
    rows_per_batch = 100
    table = pa.table(
        {
            "id": list(range(n_batches * rows_per_batch)),
            "payload": pa.array(
                [b"x" * 1024] * (n_batches * rows_per_batch), type=pa.large_binary()
            ),
        }
    )

    ds = CachedDataset(
        table.to_reader(max_chunksize=rows_per_batch),
        memory_capacity=32 * 1024,  # 32 KiB — forces spill to disk
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    ds.ingest_all()

    disk_files = list(tmp_path.rglob("*"))
    assert len(disk_files) > 0, "Expected Foyer to write cache files before close()"

    ds.close()

    remaining = list(tmp_path.rglob("*"))
    assert remaining == [], (
        f"Expected disk_path to be empty after close(), found: {remaining}"
    )


def test_drop_removes_disk_files(tmp_path):
    """Drop (when dataset goes out of scope) must delete the Foyer block-device files it wrote to disk."""
    n_batches = 5
    rows_per_batch = 100
    table = pa.table(
        {
            "id": list(range(n_batches * rows_per_batch)),
            "payload": pa.array(
                [b"x" * 1024] * (n_batches * rows_per_batch), type=pa.large_binary()
            ),
        }
    )

    ds = CachedDataset(
        table.to_reader(max_chunksize=rows_per_batch),
        memory_capacity=32 * 1024,  # 32 KiB — forces spill to disk
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    ds.ingest_all()

    disk_files = list(tmp_path.rglob("*"))
    assert len(disk_files) > 0, "Expected Foyer to write cache files before drop()"

    # Let ds go out of scope (drop is called automatically)
    del ds

    # Check that the directory no longer exists or is empty
    assert not tmp_path.exists() or list(tmp_path.rglob("*")) == [], (
        f"Expected disk_path to be removed or empty after drop(), found: {list(tmp_path.rglob('*')) if tmp_path.exists() else 'directory does not exist'}"
    )
