import duckdb
import pyarrow as pa

from batchcorder import CachedDataset


EMPLOYEE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("name", pa.string()),
        ("manager_id", pa.int64()),  # null for root employees
    ]
)

EMPLOYEE_BATCHES = [
    pa.record_batch(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "manager_id": [None, 1, 1],
        },
        schema=EMPLOYEE_SCHEMA,
    ),
    pa.record_batch(
        {"id": [4, 5, 6], "name": ["Dave", "Eve", "Frank"], "manager_id": [2, 2, 3]},
        schema=EMPLOYEE_SCHEMA,
    ),
]

LARGE_SCHEMA = pa.schema([("id", pa.int64()), ("value", pa.float64())])
LARGE_ROWS_PER_BATCH = 100
LARGE_TOTAL_BATCHES = 10
LARGE_TOTAL_ROWS = LARGE_ROWS_PER_BATCH * LARGE_TOTAL_BATCHES  # 1 000


def _large_batches() -> list[pa.RecordBatch]:
    return [
        pa.record_batch(
            {
                "id": list(
                    range(i * LARGE_ROWS_PER_BATCH, (i + 1) * LARGE_ROWS_PER_BATCH)
                ),
                "value": [
                    float(j)
                    for j in range(
                        i * LARGE_ROWS_PER_BATCH, (i + 1) * LARGE_ROWS_PER_BATCH
                    )
                ],
            },
            schema=LARGE_SCHEMA,
        )
        for i in range(LARGE_TOTAL_BATCHES)
    ]


class CountingReader:
    """Wraps a batch list and counts how many batches the upstream has yielded.

    Implements ``__arrow_c_stream__`` via a lazy generator so the counter
    increments at the C-stream level as each batch is actually consumed.
    """

    def __init__(self, batches: list[pa.RecordBatch], schema: pa.Schema) -> None:
        self._batches = batches
        self._schema = schema
        self.batches_read = 0

    def __arrow_c_stream__(self, requested_schema=None):
        def _gen():
            for batch in self._batches:
                self.batches_read += 1
                yield batch

        return pa.RecordBatchReader.from_batches(
            self._schema, _gen()
        ).__arrow_c_stream__(requested_schema)


def test_self_join(tmp_path):
    """CachedDataset: self-join returns the correct manager mapping.
    Bare reader: self-join silently returns zero rows.

    DuckDB calls __arrow_c_stream__ once per side of the join.  CachedDataset
    replays from the cache on each call; a bare reader is exhausted after the
    first scan, so the second side sees nothing and the join produces no rows.
    """
    source = CountingReader(EMPLOYEE_BATCHES, EMPLOYEE_SCHEMA)
    ds = CachedDataset(source, 16 * 1024 * 1024, str(tmp_path), 64 * 1024 * 1024)

    con = duckdb.connect()
    con.register("employees", ds)

    rows = con.execute("""
        SELECT e.name AS employee, m.name AS manager
        FROM employees e
        JOIN employees m ON e.manager_id = m.id
        ORDER BY e.name
    """).fetchall()

    # CachedDataset: correct result
    assert rows == [
        ("Bob", "Alice"),
        ("Carol", "Alice"),
        ("Dave", "Bob"),
        ("Eve", "Bob"),
        ("Frank", "Carol"),
    ]
    assert "Alice" not in {r[0] for r in rows}  # root employee excluded

    # Source was ingested exactly once regardless of how many times DuckDB scanned
    assert source.batches_read == len(EMPLOYEE_BATCHES)
    assert ds.ingested_count == len(EMPLOYEE_BATCHES)
    assert ds.upstream_exhausted is True

    # Bare reader: same query returns empty because the stream is single-use
    bare = pa.RecordBatchReader.from_batches(EMPLOYEE_SCHEMA, iter(EMPLOYEE_BATCHES))
    con_bare = duckdb.connect()
    con_bare.register("employees", bare)

    bare_rows = con_bare.execute("""
        SELECT e.name AS employee, m.name AS manager
        FROM employees e
        JOIN employees m ON e.manager_id = m.id
        ORDER BY e.name
    """).fetchall()

    assert not bare_rows


def test_limit_does_not_exhaust_upstream(tmp_path):
    batches = _large_batches()

    ds = CachedDataset(
        pa.RecordBatchReader.from_batches(LARGE_SCHEMA, iter(batches)),
        memory_capacity=64 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=256 * 1024 * 1024,
    )
    con = duckdb.connect()
    con.register("data", ds)

    limited = con.execute("SELECT * FROM data LIMIT 50").fetchall()
    assert len(limited) == 50

    # A follow-up full scan still gets every row (cache + remaining upstream)
    row = con.execute("SELECT COUNT(*) FROM data").fetchone()
    assert row is not None
    total = row[0]
    assert total == LARGE_TOTAL_ROWS
    assert ds.upstream_exhausted is True

    bare = pa.RecordBatchReader.from_batches(LARGE_SCHEMA, iter(batches))
    con_bare = duckdb.connect()
    con_bare.register("data", bare)

    bare_limited = con_bare.execute("SELECT * FROM data LIMIT 50").fetchall()
    assert len(bare_limited) == 50

    # After the LIMIT query the stream is partially spent; the second query
    # cannot recover the discarded batches and sees fewer than LARGE_TOTAL_ROWS.
    bare_row = con_bare.execute("SELECT COUNT(*) FROM data").fetchone()
    assert bare_row is not None
    (bare_total,) = bare_row
    assert bare_total < LARGE_TOTAL_ROWS


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
                [
                    1731751215500000,  # 2024-11-16 12:00:15.500
                    1731751215700000,  # 2024-11-16 12:00:15.700
                ],
                type=pa.timestamp("us"),
            ),
        },
        schema=_SENSOR_SCHEMA,
    ),
    pa.record_batch(
        {
            "site": ["a", "b", "a"],
            "humidity": [0.5, 0.6, 0.7],
            "event_time": pa.array(
                [
                    1731866534950000,  # 2024-11-17 18:12:14.950
                    1731866535120000,  # 2024-11-17 18:12:15.120
                    1731952935100000,  # 2024-11-18 18:12:15.100
                ],
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
                [
                    1731751215400000,  # 2024-11-16 12:00:15.400
                    1731866535100000,  # 2024-11-17 18:12:15.100
                    1731952935100000,  # 2024-11-18 18:12:15.100
                ],
                type=pa.timestamp("us"),
            ),
        },
        schema=_EVENT_SCHEMA,
    ),
]

_ASOF_SQL = """
    SELECT s.site, s.humidity, e.event_type
    FROM sensors s
    ASOF LEFT JOIN events e
        ON s.site = e.site AND s.event_time >= e.event_time
    ORDER BY s.event_time
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


def _reference_result(sql: str) -> list:
    """Run *sql* against full in-memory PyArrow tables to get the ground-truth result."""
    sensors_table = pa.Table.from_batches(_SENSOR_BATCHES)
    events_table = pa.Table.from_batches(_EVENT_BATCHES)
    con = duckdb.connect()
    con.register("sensors", sensors_table)
    con.register("events", events_table)
    return con.execute(sql).fetchall()


def _make_asof_con(sensors_source, events_source) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("sensors", sensors_source)
    con.register("events", events_source)
    return con


def test_asof_join_without_tolerance(tmp_path):
    sensors_ds = CachedDataset(
        pa.RecordBatchReader.from_batches(_SENSOR_SCHEMA, iter(_SENSOR_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "sensors"),
        disk_capacity=64 * 1024 * 1024,
    )
    events_ds = CachedDataset(
        pa.RecordBatchReader.from_batches(_EVENT_SCHEMA, iter(_EVENT_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "events"),
        disk_capacity=64 * 1024 * 1024,
    )
    expected = _reference_result(_ASOF_SQL)
    rows = _make_asof_con(sensors_ds, events_ds).execute(_ASOF_SQL).fetchall()
    assert rows == expected

    # Bare readers also work: a simple ASOF JOIN scans each table exactly once.
    bare_sensors = pa.RecordBatchReader.from_batches(
        _SENSOR_SCHEMA, iter(_SENSOR_BATCHES)
    )
    bare_events = pa.RecordBatchReader.from_batches(_EVENT_SCHEMA, iter(_EVENT_BATCHES))
    bare_rows = _make_asof_con(bare_sensors, bare_events).execute(_ASOF_SQL).fetchall()
    assert bare_rows == expected


def test_asof_join_with_tolerance(tmp_path):
    sensors_ds = CachedDataset(
        pa.RecordBatchReader.from_batches(_SENSOR_SCHEMA, iter(_SENSOR_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "sensors"),
        disk_capacity=64 * 1024 * 1024,
    )
    events_ds = CachedDataset(
        pa.RecordBatchReader.from_batches(_EVENT_SCHEMA, iter(_EVENT_BATCHES)),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path / "events"),
        disk_capacity=64 * 1024 * 1024,
    )
    expected = _reference_result(_ASOF_TOLERANCE_SQL)
    rows = _make_asof_con(sensors_ds, events_ds).execute(_ASOF_TOLERANCE_SQL).fetchall()
    assert rows == expected

    bare_sensors = pa.RecordBatchReader.from_batches(
        _SENSOR_SCHEMA, iter(_SENSOR_BATCHES)
    )
    bare_events = pa.RecordBatchReader.from_batches(_EVENT_SCHEMA, iter(_EVENT_BATCHES))
    bare_rows = (
        _make_asof_con(bare_sensors, bare_events)
        .execute(_ASOF_TOLERANCE_SQL)
        .fetchall()
    )
    # The stream is partially or fully drained by one branch of the query before
    # the other branch runs.  The exact row count depends on DuckDB's scheduler,
    # but in all cases the result is wrong: no events are matched.
    assert bare_rows != expected
    assert all(row[2] is None for row in bare_rows)
