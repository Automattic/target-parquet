import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal
from singer_sdk.helpers._flattening import flatten_schema

from target_parquet.utils.parquet import (
    EXTENSION_MAPPING,
    _field_type_to_pyarrow_field,
    concat_tables,
    create_pyarrow_table,
    flatten_schema_to_pyarrow_schema,
    get_pyarrow_table_size,
    write_parquet_file,
)


@pytest.fixture()
def sample_data():
    return [
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 30},
        {"id": 3, "name": "Charlie", "age": 22},
    ]


@pytest.fixture()
def sample_schema():
    return pa.schema(
        [
            ("id", pa.int64()),
            ("name", pa.string()),
            ("age", pa.int64()),
        ]
    )


def test_flatten_schema_to_pyarrow_schema():
    schema = {
        "type": "object",
        "properties": {
            "str": {"type": ["null", "string"]},
            "int": {"type": ["null", "integer"]},
            "decimal": {"type": ["null", "number"]},
            "decimal2": {"type": ["null", "number"]},
            "date": {"type": ["null", "string"], "format": "date-time"},
            "datetime": {"type": ["null", "string"], "format": "date-time"},
            "boolean": {"type": ["null", "boolean"]},
        },
    }
    flatten_schema_result = flatten_schema(schema, max_level=20)
    pyarrow_schema = flatten_schema_to_pyarrow_schema(flatten_schema_result)
    expected_pyarrow_schema = pa.schema(
        [
            pa.field("str", pa.string()),
            pa.field("int", pa.int64()),
            pa.field("decimal", pa.float64()),
            pa.field("decimal2", pa.float64()),
            pa.field("date", pa.string()),
            pa.field("datetime", pa.string()),
            pa.field("boolean", pa.bool_()),
        ]
    )
    assert pyarrow_schema == expected_pyarrow_schema


@pytest.mark.parametrize(
    "field_name, input_types, expected_result",
    [
        pytest.param(
            "example_field",
            {"type": "string"},
            pa.field("example_field", pa.string(), True),
            id="valid_input",
        ),
        pytest.param(
            "example_field_anyof",
            {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            pa.field("example_field_anyof", pa.int64(), False),
            id="anyof_input",
        ),
        pytest.param(
            "unknown_type",
            {"type": "unknown_type"},
            pa.field("unknown_type", pa.string(), True),
            id="unknown_type",
        ),
    ],
)
def test_field_type_to_pyarrow_field(field_name, input_types, expected_result):
    result = _field_type_to_pyarrow_field(
        field_name, input_types, ["example_field_anyof"]
    )
    assert result == expected_result


def test_create_pyarrow_table(sample_schema):
    data = [
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob"},
        {"id": 3, "age": 22},
    ]
    expected_table = pd.DataFrame(data)
    result_table = create_pyarrow_table(data, sample_schema)

    # Check if the result has the expected schema
    assert result_table.schema.equals(sample_schema)
    # Check if the result has the expected number of rows
    assert len(result_table) == len(data)
    # Check if the result has the expected data
    assert result_table.to_pandas().equals(expected_table)


def test_concat_tables(sample_data, sample_schema):
    # Define the initial PyArrow schema and table
    initial_table = create_pyarrow_table(sample_data, sample_schema)

    # Call concat_tables with sample data
    result_table = concat_tables(sample_data, initial_table, sample_schema)

    # Create the expected PyArrow table using create_pyarrow_table
    expected_table = create_pyarrow_table(sample_data * 2, sample_schema)

    # Check if the resulting PyArrow table is equal to the expected table
    assert result_table.equals(expected_table)


@pytest.mark.parametrize("compression_method", ["gzip", "snappy"])
@pytest.mark.parametrize("partition_cols", [None, ["name"]])
def test_write_parquet_file(
    tmpdir, sample_data, sample_schema, compression_method, partition_cols
):
    # Create a PyArrow table from sample data
    table = create_pyarrow_table(sample_data, sample_schema)

    # Define the path for the Parquet file within the temporary directory
    parquet_path = tmpdir.mkdir("test_parquet_file")

    # Test writing to Parquet file with different compression methods and partition columns
    write_parquet_file(
        table,
        str(parquet_path),
        basename_template="test_parquet_file-{i}",
        compression_method=compression_method,
        partition_cols=partition_cols,
    )

    # Check if the Parquet file was created
    file_name = f"test_parquet_file-0{EXTENSION_MAPPING[compression_method]}.parquet"
    expected_table = pd.DataFrame(sample_data)
    if partition_cols:
        # Change the file name to include the partition column value and filter the expected data
        file_name = os.path.join("name=Alice", file_name)
        expected_table = expected_table.loc[expected_table["name"] == "Alice"]
        expected_table["name"] = expected_table["name"].astype("category")

    assert parquet_path.join(file_name).check()

    # Check if the Parquet file contains the expected data
    read_table = pq.read_table(str(parquet_path.join(file_name)))

    assert_frame_equal(
        read_table.to_pandas(),
        expected_table,
        check_like=True,  # ignore the order of index & columns
    )


def test_get_pyarrow_table_size(sample_data, sample_schema):
    # Create a PyArrow table with sample data
    table = create_pyarrow_table(sample_data * 100000, sample_schema)

    # Test the get_pyarrow_table_size function
    size_in_mb = get_pyarrow_table_size(table)

    # Check if the result is a non-negative float
    assert isinstance(size_in_mb, float)
    assert pytest.approx(size_in_mb, 0.1) == 7.15
