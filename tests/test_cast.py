"""Tests for StreamCache.cast and StreamCacheReader.cast.

Key property under test
-----------------------
* ``StreamCache.cast(schema)`` returns a **replayable** ``CastingStreamCache``
  whose ``__arrow_c_stream__`` can be called multiple times.  This mirrors the
  multi-scan behaviour of ``StreamCache`` itself and allows DuckDB self-joins
  and ASOF joins to produce correct results — the same guarantee that
  ``StreamCache`` provides for uncast data.

* ``StreamCacheReader.cast(schema)`` returns a one-shot
  ``pa.RecordBatchReader``, consistent with the one-shot semantics of
  ``StreamCacheReader``.
"""

import duckdb
import pyarrow as pa
import pytest

from batchcorder import CastingStreamCache, StreamCache


def _ds(tmp_path, table, batch_size=3):
    return StreamCache(
        table.to_reader(max_chunksize=batch_size),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )


TABLE = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5, 6], type=pa.int32()),
        "label": pa.array(["a", "b", "c", "d", "e", "f"], type=pa.utf8()),
        "score": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type=pa.float32()),
    }
)

TARGET_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("label", pa.large_utf8()),
        pa.field("score", pa.float64()),
    ]
)


def test_dataset_cast_returns_casting_dataset(tmp_path):
    """cast() on StreamCache returns a CastingStreamCache, not a RecordBatchReader."""
    ds = _ds(tmp_path, TABLE)
    result = ds.cast(TARGET_SCHEMA)
    assert isinstance(result, CastingStreamCache)


def test_dataset_cast_schema_property(tmp_path):
    """CastingStreamCache.schema reflects the target schema."""
    ds = _ds(tmp_path, TABLE)
    casting = ds.cast(TARGET_SCHEMA)
    schema = pa.schema(casting.schema)
    assert schema.field("id").type == pa.int64()
    assert schema.field("label").type == pa.large_utf8()
    assert schema.field("score").type == pa.float64()


def test_dataset_cast_applies_schema(tmp_path):
    """CastingStreamCache produces batches with the target schema."""
    ds = _ds(tmp_path, TABLE)
    result = pa.table(ds.cast(TARGET_SCHEMA))
    assert result.schema.field("id").type == pa.int64()
    assert result.schema.field("label").type == pa.large_utf8()
    assert result.schema.field("score").type == pa.float64()


def test_dataset_cast_preserves_values(tmp_path):
    """CastingStreamCache does not alter row values, only types."""
    result = pa.table(_ds(tmp_path, TABLE).cast(TARGET_SCHEMA))
    assert result.column("id").to_pylist() == TABLE.column("id").to_pylist()
    assert result.column("label").to_pylist() == TABLE.column("label").to_pylist()
    assert result.column("score").to_pylist() == TABLE.column("score").to_pylist()


def test_dataset_cast_multi_batch(tmp_path):
    """CastingStreamCache works correctly when the stream spans multiple batches."""
    casting = _ds(tmp_path, TABLE, batch_size=2).cast(TARGET_SCHEMA)
    result = pa.table(casting)
    assert result.num_rows == TABLE.num_rows
    assert result.schema.field("id").type == pa.int64()


def test_dataset_cast_noop_same_schema(tmp_path):
    """CastingStreamCache with the identical schema returns equivalent data."""
    result = pa.table(_ds(tmp_path, TABLE).cast(TABLE.schema))
    assert result.equals(TABLE)


def test_dataset_cast_is_replayable_multiple_reads(tmp_path):
    """CastingStreamCache can be read multiple times; each read returns correct data."""
    casting = _ds(tmp_path, TABLE).cast(TARGET_SCHEMA)
    r1 = pa.RecordBatchReader.from_stream(casting).read_all()
    r2 = pa.RecordBatchReader.from_stream(casting).read_all()
    r3 = pa.RecordBatchReader.from_stream(casting).read_all()
    assert r1.equals(r2)
    assert r2.equals(r3)


def test_dataset_cast_is_replayable_repeated_cast(tmp_path):
    """cast() can be called repeatedly on the same StreamCache."""
    ds = _ds(tmp_path, TABLE)
    r1 = pa.table(ds.cast(TARGET_SCHEMA))
    r2 = pa.table(ds.cast(TARGET_SCHEMA))
    assert r1.equals(r2)


def test_dataset_cast_bare_reader_is_not_replayable(tmp_path):
    """Control: a bare pa.RecordBatchReader is one-shot and cannot be replayed."""
    bare = pa.RecordBatchReader.from_stream(_ds(tmp_path, TABLE).cast(TARGET_SCHEMA))
    first = bare.read_all()
    assert first.num_rows == TABLE.num_rows
    second = bare.read_all()  # stream exhausted — returns empty table
    assert second.num_rows == 0


_EMPLOYEE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("name", pa.string()),
        ("manager_id", pa.int64()),
    ]
)
_EMPLOYEE_BATCHES = [
    pa.record_batch(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "manager_id": [None, 1, 1],
        },
        schema=_EMPLOYEE_SCHEMA,
    ),
    pa.record_batch(
        {"id": [4, 5, 6], "name": ["Dave", "Eve", "Frank"], "manager_id": [2, 2, 3]},
        schema=_EMPLOYEE_SCHEMA,
    ),
]

# Cast name to large_utf8; join key columns (id, manager_id) stay int64.
_CAST_EMPLOYEE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("name", pa.large_utf8()),
        ("manager_id", pa.int64()),
    ]
)

_SELF_JOIN_SQL = """
    SELECT e.name AS employee, m.name AS manager
    FROM employees e
    JOIN employees m ON e.manager_id = m.id
    ORDER BY e.name
"""

_EXPECTED_SELF_JOIN = [
    ("Bob", "Alice"),
    ("Carol", "Alice"),
    ("Dave", "Bob"),
    ("Eve", "Bob"),
    ("Frank", "Carol"),
]


def _make_employee_ds(tmp_path):
    return StreamCache(
        pa.RecordBatchReader.from_batches(_EMPLOYEE_SCHEMA, iter(_EMPLOYEE_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )


def test_casting_dataset_self_join_matches_uncast(tmp_path):
    """CastingStreamCache self-join produces the same result as an uncast self-join.

    DuckDB calls ``__arrow_c_stream__`` once per side of the join.
    ``CastingStreamCache`` creates a fresh reader on each call (replayable), so
    both sides see the full dataset and the join returns correct rows.
    A bare ``pa.RecordBatchReader`` would be exhausted after the first side
    and return zero rows.
    """
    ds = _make_employee_ds(tmp_path / "uncast")
    casting = ds.cast(_CAST_EMPLOYEE_SCHEMA)

    con = duckdb.connect()
    con.register("employees", casting)
    rows = con.execute(_SELF_JOIN_SQL).fetchall()

    assert rows == _EXPECTED_SELF_JOIN

    # Cross-check: uncast StreamCache produces the same result.
    ds2 = _make_employee_ds(tmp_path / "plain")
    con2 = duckdb.connect()
    con2.register("employees", ds2)
    plain_rows = con2.execute(_SELF_JOIN_SQL).fetchall()
    assert rows == plain_rows


def test_casting_dataset_self_join_bare_reader_fails(tmp_path):
    """Control: a bare RecordBatchReader cast result fails the self-join.

    ``StreamCacheReader.cast`` returns a one-shot ``pa.RecordBatchReader``.
    DuckDB exhausts it on the first side of the join; the second side sees
    nothing, so the join returns zero rows.
    """
    ds = _make_employee_ds(tmp_path)
    bare_cast = ds.reader().cast(_CAST_EMPLOYEE_SCHEMA)  # one-shot

    con = duckdb.connect()
    con.register("employees", bare_cast)
    rows = con.execute(_SELF_JOIN_SQL).fetchall()

    assert rows != _EXPECTED_SELF_JOIN  # wrong — stream exhausted after first side
    assert len(rows) == 0


_SENSOR_SCHEMA = pa.schema(
    [
        ("site", pa.string()),
        ("humidity", pa.float64()),
        ("event_time", pa.timestamp("us")),
    ]
)
_SENSOR_BATCHES = [
    pa.record_batch(
        {
            "site": ["a", "b"],
            "humidity": [0.3, 0.4],
            "event_time": pa.array(
                [1731751215500000, 1731751215700000], type=pa.timestamp("us")
            ),
        },
        schema=_SENSOR_SCHEMA,
    ),
    pa.record_batch(
        {
            "site": ["a", "b", "a"],
            "humidity": [0.5, 0.6, 0.7],
            "event_time": pa.array(
                [1731866534950000, 1731866535120000, 1731952935100000],
                type=pa.timestamp("us"),
            ),
        },
        schema=_SENSOR_SCHEMA,
    ),
]

_EVENT_SCHEMA = pa.schema(
    [
        ("site", pa.string()),
        ("event_type", pa.string()),
        ("event_time", pa.timestamp("us")),
    ]
)
_EVENT_BATCHES = [
    pa.record_batch(
        {
            "site": ["a", "b", "a"],
            "event_type": ["cloud coverage", "rain start", "rain stop"],
            "event_time": pa.array(
                [1731751215400000, 1731866535100000, 1731952935100000],
                type=pa.timestamp("us"),
            ),
        },
        schema=_EVENT_SCHEMA,
    ),
]

# Cast string → large_utf8 while keeping timestamps unchanged.
_CAST_SENSOR_SCHEMA = pa.schema(
    [
        ("site", pa.large_utf8()),
        ("humidity", pa.float64()),
        ("event_time", pa.timestamp("us")),
    ]
)
_CAST_EVENT_SCHEMA = pa.schema(
    [
        ("site", pa.large_utf8()),
        ("event_type", pa.large_utf8()),
        ("event_time", pa.timestamp("us")),
    ]
)

_ASOF_SQL = """
    SELECT s.site, s.humidity, e.event_type
    FROM sensors s
    ASOF LEFT JOIN events e
        ON s.site = e.site AND s.event_time >= e.event_time
    ORDER BY s.event_time
"""


def _reference_result() -> list:
    """Ground truth from fully in-memory PyArrow tables."""
    sensors_table = pa.Table.from_batches(_SENSOR_BATCHES)
    events_table = pa.Table.from_batches(_EVENT_BATCHES)
    con = duckdb.connect()
    con.register("sensors", sensors_table)
    con.register("events", events_table)
    return con.execute(_ASOF_SQL).fetchall()


def test_casting_dataset_asof_join_matches_uncast(tmp_path):
    """CastingStreamCache ASOF join returns correct rows, same as uncast StreamCache.

    An ASOF join scans each side once, so even a one-shot reader would work.
    This test confirms that the cast does not corrupt values and that the
    CastingStreamCache is a valid Arrow stream source for DuckDB.
    """
    sensors_ds = StreamCache(
        pa.RecordBatchReader.from_batches(_SENSOR_SCHEMA, iter(_SENSOR_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "sensors"),
        disk_capacity=64 * 1024 * 1024,
    )
    events_ds = StreamCache(
        pa.RecordBatchReader.from_batches(_EVENT_SCHEMA, iter(_EVENT_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "events"),
        disk_capacity=64 * 1024 * 1024,
    )

    cast_sensors = sensors_ds.cast(_CAST_SENSOR_SCHEMA)
    cast_events = events_ds.cast(_CAST_EVENT_SCHEMA)

    con = duckdb.connect()
    con.register("sensors", cast_sensors)
    con.register("events", cast_events)
    rows = con.execute(_ASOF_SQL).fetchall()

    expected = _reference_result()
    assert rows == expected


def test_casting_dataset_asof_join_with_tolerance_replayable(tmp_path):
    """CastingStreamCache is replayable under the tolerance ASOF join pattern.

    The tolerance ASOF query re-scans the sensors table (ASOF JOIN + a second
    LEFT JOIN on the same source).  A bare reader would be exhausted on the
    first scan and return wrong rows; a CastingStreamCache replays from cache and
    returns the correct result — matching the uncast StreamCache behaviour.
    """
    _ASOF_TOLERANCE_SQL = """
        WITH matched AS (
            SELECT s.site, s.event_time, e.event_type
            FROM sensors s
            ASOF LEFT JOIN events e
                ON s.site = e.site AND s.event_time >= e.event_time
            WHERE e.event_time IS NULL
               OR s.event_time - e.event_time <= INTERVAL 1 SECOND
        )
        SELECT s.site, s.humidity, m.event_type
        FROM sensors s
        LEFT JOIN matched m ON s.site = m.site AND s.event_time = m.event_time
        ORDER BY s.event_time
    """

    def _reference(sql):
        sensors_table = pa.Table.from_batches(_SENSOR_BATCHES)
        events_table = pa.Table.from_batches(_EVENT_BATCHES)
        con = duckdb.connect()
        con.register("sensors", sensors_table)
        con.register("events", events_table)
        return con.execute(sql).fetchall()

    sensors_ds = StreamCache(
        pa.RecordBatchReader.from_batches(_SENSOR_SCHEMA, iter(_SENSOR_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "sensors"),
        disk_capacity=64 * 1024 * 1024,
    )
    events_ds = StreamCache(
        pa.RecordBatchReader.from_batches(_EVENT_SCHEMA, iter(_EVENT_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "events"),
        disk_capacity=64 * 1024 * 1024,
    )

    cast_sensors = sensors_ds.cast(_CAST_SENSOR_SCHEMA)
    cast_events = events_ds.cast(_CAST_EVENT_SCHEMA)

    con = duckdb.connect()
    con.register("sensors", cast_sensors)
    con.register("events", cast_events)
    rows = con.execute(_ASOF_TOLERANCE_SQL).fetchall()

    expected = _reference(_ASOF_TOLERANCE_SQL)
    assert rows == expected


def test_reader_cast_returns_record_batch_reader(tmp_path):
    """cast() on StreamCacheReader returns a pa.RecordBatchReader (one-shot)."""
    reader = _ds(tmp_path, TABLE).reader()
    result = reader.cast(TARGET_SCHEMA)
    assert isinstance(result, pa.RecordBatchReader)


def test_reader_cast_applies_schema(tmp_path):
    """cast() on a reader produces batches with the target schema."""
    result = _ds(tmp_path, TABLE).reader().cast(TARGET_SCHEMA).read_all()
    assert result.schema.field("id").type == pa.int64()
    assert result.schema.field("label").type == pa.large_utf8()
    assert result.schema.field("score").type == pa.float64()


def test_reader_cast_preserves_values(tmp_path):
    """cast() on a reader does not alter row values."""
    result = _ds(tmp_path, TABLE).reader().cast(TARGET_SCHEMA).read_all()
    assert result.column("id").to_pylist() == TABLE.column("id").to_pylist()
    assert result.column("label").to_pylist() == TABLE.column("label").to_pylist()
    assert result.column("score").to_pylist() == TABLE.column("score").to_pylist()


def test_reader_cast_consumes_reader(tmp_path):
    """Calling cast() marks the reader as consumed."""
    reader = _ds(tmp_path, TABLE).reader()
    assert not reader.closed
    reader.cast(TARGET_SCHEMA).read_all()
    assert reader.closed


def test_reader_cast_raises_if_already_consumed(tmp_path):
    """cast() on an already-consumed reader raises ValueError."""
    reader = _ds(tmp_path, TABLE).reader()
    reader.cast(TARGET_SCHEMA).read_all()
    with pytest.raises(ValueError, match="already consumed"):
        reader.cast(TARGET_SCHEMA)


def test_reader_cast_multi_batch(tmp_path):
    """cast() on reader works correctly across multiple batches."""
    result = _ds(tmp_path, TABLE, batch_size=2).reader().cast(TARGET_SCHEMA).read_all()
    assert result.num_rows == TABLE.num_rows
    assert result.schema.field("id").type == pa.int64()


def test_reader_cast_noop_same_schema(tmp_path):
    """cast() with the same schema returns equivalent data."""
    result = _ds(tmp_path, TABLE).reader().cast(TABLE.schema).read_all()
    assert result.equals(TABLE)
