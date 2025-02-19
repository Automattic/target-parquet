import json
import os.path
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from singer_sdk import typing as th
from singer_sdk.testing import target_sync_test

from target_parquet.target import TargetParquet


@pytest.fixture(scope="session")
def test_output_dir():
    return Path(f".output/test_{uuid4()}/")


@pytest.fixture(scope="session")
def sample_config(test_output_dir):
    return {
        "destination_path": str(test_output_dir),
    }


@pytest.fixture()
def example1_schema_messages():
    stream_name = f"test_schema_{str(uuid4()).split('-')[-1]}"
    schema_message = {
        "type": "SCHEMA",
        "stream": stream_name,
        "schema": {
            "type": "object",
            "properties": {"col_a": th.StringType().to_dict()},
        },
    }
    tap_output = "\n".join(
        json.dumps(msg)
        for msg in [
            schema_message,
            {
                "type": "RECORD",
                "stream": stream_name,
                "record": {"col_a": "samplerow1"},
            },
            {
                "type": "RECORD",
                "stream": stream_name,
                "record": {"col_a": "samplerow2"},
            },
        ]
    )
    return {
        "stream_name": stream_name,
        "schema": schema_message,
        "messages": tap_output,
    }


@pytest.fixture()
def example2_schema_messages_mixed():
    stream_name = f"test_schema_{str(uuid4()).split('-')[-1]}"
    schema_message1 = {
        "type": "SCHEMA",
        "stream": f"{stream_name}_A",
        "schema": {
            "type": "object",
            "properties": {"col_a": th.StringType().to_dict()},
        },
    }
    schema_message2 = {
        "type": "SCHEMA",
        "stream": f"{stream_name}_B",
        "schema": {
            "type": "object",
            "properties": {"col_b": th.StringType().to_dict()},
        },
    }
    tap_output = "\n".join(
        json.dumps(msg)
        for msg in [
            schema_message1,
            {
                "type": "RECORD",
                "stream": f"{stream_name}_A",
                "record": {"col_a": "samplerow1"},
            },
            schema_message2,
            {
                "type": "RECORD",
                "stream": f"{stream_name}_B",
                "record": {"col_b": "samplerow2"},
            },
        ]
        * 3
    )
    return {
        "stream_names": [f"{stream_name}_A", f"{stream_name}_B"],
        "schemas": [schema_message1, schema_message2],
        "messages": tap_output,
    }


def test_e2e_create_file(
        monkeypatch, test_output_dir, sample_config, example1_schema_messages
):
    """Test that the target creates a file with the expected records"""
    monkeypatch.setattr("time.time", lambda: 1700000000)

    target_sync_test(
        TargetParquet(config=sample_config),
        input=StringIO(example1_schema_messages["messages"]),
        finalize=True,
    )

    assert (
            len(os.listdir(test_output_dir / example1_schema_messages["stream_name"])) == 1
    )

    expected = pd.DataFrame({"col_a": ["samplerow1", "samplerow2"]})
    result = pd.read_parquet(test_output_dir / example1_schema_messages["stream_name"])
    assert expected.equals(result)


def test_e2e_create_file_mixed_schemas(
        monkeypatch, test_output_dir, sample_config, example2_schema_messages_mixed
):
    """Test that the target creates a file for each schema if we have mixed schemas"""
    monkeypatch.setattr("time.time", lambda: 1700000000)

    target_sync_test(
        TargetParquet(config=sample_config),
        input=StringIO(example2_schema_messages_mixed["messages"]),
        finalize=True,
    )

    assert (
            len(
                os.listdir(
                    test_output_dir / example2_schema_messages_mixed["stream_names"][0]
                )
            )
            == 1
    )
    assert (
            len(
                os.listdir(
                    test_output_dir / example2_schema_messages_mixed["stream_names"][1]
                )
            )
            == 1
    )

    expected = pd.DataFrame({"col_a": ["samplerow1"] * 3})
    result = pd.read_parquet(
        test_output_dir / example2_schema_messages_mixed["stream_names"][0]
    )
    assert expected.equals(result)

    expected = pd.DataFrame({"col_b": ["samplerow2"] * 3})
    result = pd.read_parquet(
        test_output_dir / example2_schema_messages_mixed["stream_names"][1]
    )
    assert expected.equals(result)


def test_e2e_create_file_with_null_fields(monkeypatch, test_output_dir, sample_config):
    """This tests checks if the null object fields are being correctly exploded according to the schema and
    if it doesn't replace the values if we have a conflict of the same field name in different levels of object.
    """
    monkeypatch.setattr("time.time", lambda: 1700000000)
    stream_name = f"test_schema_{str(uuid4()).split('-')[-1]}"
    schema_message = {
        "type": "SCHEMA",
        "stream": stream_name,
        "schema": {
            "type": "object",
            "properties": {
                "field1": {
                    "type": ["null", "object"],
                    "additionalProperties": False,
                    "properties": {
                        "field2": {
                            "type": ["null", "object"],
                            "properties": {
                                "field3": {"type": ["null", "string"]},
                                "field4": {"type": ["null", "string"]},
                            },
                        }
                    },
                },
                "field2": {
                    "type": ["null", "object"],
                    "properties": {
                        "field3": {"type": ["null", "string"]},
                        "field4": {"type": ["null", "string"]},
                        "field5": {"type": ["null", "string"]},
                    },
                },
                "field6": {"type": ["null", "string"]},
            },
        },
    }
    tap_output = "\n".join(
        json.dumps(msg)
        for msg in [
            schema_message,
            {
                "type": "RECORD",
                "stream": stream_name,
                "record": {
                    "field1": {
                        "field2": {"field3": "test_field3", "field4": "test_field4"}
                    },
                    "field2": None,
                },
            },
        ]
    )

    target_sync_test(
        TargetParquet(config=sample_config),
        input=StringIO(tap_output),
        finalize=True,
    )

    expected_file_name = stream_name + '-20231114_221320-0-0.gz.parquet'
    assert os.listdir(test_output_dir / stream_name)[0] == expected_file_name

    expected = pd.DataFrame(
        [
            {
                "field1__field2__field3": "test_field3",
                "field1__field2__field4": "test_field4",
                "field2__field3": None,
                "field2__field4": None,
                "field2__field5": None,
                "field6": None,
            }
        ]
    )
    result = pd.read_parquet(test_output_dir / stream_name)
    assert expected.equals(result)


def test_e2e_more_records_than_batch_size(monkeypatch, test_output_dir, sample_config):
    """Test that the target creates a file with the expected number of records while having more records than the batch size"""
    monkeypatch.setattr("time.time", lambda: 1700000000)
    stream_name = f"test_schema_{str(uuid4()).split('-')[-1]}"
    schema_message = {
        "type": "SCHEMA",
        "stream": stream_name,
        "schema": {
            "type": "object",
            "properties": {
                "col_a": th.StringType().to_dict(),
                "col_b": th.StringType().to_dict(),
                "col_c": th.StringType().to_dict(),
                "col_d": th.StringType().to_dict(),
                "col_e": th.StringType().to_dict(),
                "col_f": th.StringType().to_dict(),
                "col_g": th.StringType().to_dict(),
                "col_h": th.StringType().to_dict(),
            },
        },
    }
    tap_output = "\n".join(
        json.dumps(msg)
        for msg in [schema_message]
        + [
            {
                "type": "RECORD",
                "stream": stream_name,
                "record": {
                    "col_a": "samplerow1",
                    "col_b": "samplerow1",
                    "col_c": "samplerow1",
                    "col_d": "samplerow1",
                    "col_e": "samplerow1",
                    "col_f": "samplerow1",
                    "col_g": "samplerow1",
                    "col_h": "samplerow1",
                },
            }
        ]
        * 1000000
    )

    target_sync_test(
        TargetParquet(config=sample_config),
        input=StringIO(tap_output),
        finalize=True,
    )

    result = pd.read_parquet(test_output_dir / stream_name)
    assert result.shape == (1000000, 8)
    assert len(os.listdir(test_output_dir / stream_name)) == 1


@pytest.mark.parametrize(("max_pyarrow_table_size", 'files_count'),
                         [
                            pytest.param(1, 100, id="max_pyarrow_table_size=1"),
                            pytest.param(10, 10, id="max_pyarrow_table_size=10"),
                            pytest.param(1000, 1, id="max_pyarrow_table_size=100"),
                         ]
)
def test_e2e_multiple_files(monkeypatch, test_output_dir, sample_config, max_pyarrow_table_size, files_count):
    """Test that the target creates multiple files when the pyarrow file size limit is reached"""
    monkeypatch.setattr("time.time", lambda: 1700000000)
    stream_name = f"test_schema_{str(uuid4()).split('-')[-1]}"
    schema_message = {
        "type": "SCHEMA",
        "stream": stream_name,
        "schema": {
            "type": "object",
            "properties": {
                "col_a": th.StringType().to_dict(),
                "col_b": th.StringType().to_dict(),
                "col_c": th.StringType().to_dict(),
                "col_d": th.StringType().to_dict(),
                "col_e": th.StringType().to_dict(),
                "col_f": th.StringType().to_dict(),
                "col_g": th.StringType().to_dict(),
                "col_h": th.StringType().to_dict(),
            },
        },
    }
    tap_output = "\n".join(
        json.dumps(msg)
        for msg in [schema_message]
        + [
            {
                "type": "RECORD",
                "stream": stream_name,
                "record": {
                    "col_a": "samplerow1",
                    "col_b": "samplerow1",
                    "col_c": "samplerow1",
                    "col_d": "samplerow1",
                    "col_e": "samplerow1",
                    "col_f": "samplerow1",
                    "col_g": "samplerow1",
                    "col_h": "samplerow1",
                },
            }
        ]
        * 1000000
    )

    target_sync_test(
        TargetParquet(config=sample_config | {"max_pyarrow_table_size": max_pyarrow_table_size}),
        input=StringIO(tap_output),
        finalize=True,
    )

    result = pd.read_parquet(test_output_dir / stream_name)
    assert result.shape == (1000000, 8)
    assert len(os.listdir(test_output_dir / stream_name)) == files_count

def test_e2e_extra_fields(
        monkeypatch, test_output_dir, sample_config, example1_schema_messages
):
    """Test if the target can add extra fields in the record"""
    monkeypatch.setattr("time.time", lambda: 1700000000)

    target_sync_test(
        TargetParquet(
            config=sample_config
                   | {
                       "extra_fields": "field1=value1,field2=1",
                       "extra_fields_types": "field1=string,field2=integer",
                   }
        ),
        input=StringIO(example1_schema_messages["messages"]),
        finalize=True,
    )

    assert (
            len(os.listdir(test_output_dir / example1_schema_messages["stream_name"])) == 1
    )

    expected = pd.DataFrame(
        {
            "col_a": ["samplerow1", "samplerow2"],
            "field1": ["value1", "value1"],
            "field2": [1, 1],
        }
    )
    result = pd.read_parquet(test_output_dir / example1_schema_messages["stream_name"])
    assert expected.equals(result)


def test_e2e_partition_cols(
        monkeypatch, test_output_dir, sample_config, example1_schema_messages
):
    """Test if the target can partition the data"""
    monkeypatch.setattr("time.time", lambda: 1700000000)

    target_sync_test(
        TargetParquet(
            config=sample_config
                   | {
                       "extra_fields": "field1=value1",
                       "extra_fields_types": "field1=string",
                       "partition_cols": "field1",
                   }
        ),
        input=StringIO(example1_schema_messages["messages"]),
        finalize=True,
    )

    assert (
            len(
                os.listdir(
                    test_output_dir
                    / example1_schema_messages["stream_name"]
                    / "field1=value1"
                )
            )
            == 1
    )

    expected = pd.DataFrame({"col_a": ["samplerow1", "samplerow2"]})
    result = pd.read_parquet(
        test_output_dir / example1_schema_messages["stream_name"] / "field1=value1"
    )
    assert expected.equals(result)


def test_e2e_extra_fields_validation(
        monkeypatch, sample_config, example1_schema_messages
):
    """Test extra_fields and extra_fields_types validation"""
    with pytest.raises(
            AssertionError,
            match="extra_fields and extra_fields_types must be both set or both unset",
    ):
        target_sync_test(
            TargetParquet(config=sample_config | {"extra_fields": "field1=value1"}),
            input=StringIO(example1_schema_messages["messages"]),
            finalize=True,
        )

    with pytest.raises(
            AssertionError,
            match="extra_fields and extra_fields_types must be both set or both unset",
    ):
        target_sync_test(
            TargetParquet(
                config=sample_config | {"extra_fields_types": "field1=string"}
            ),
            input=StringIO(example1_schema_messages["messages"]),
            finalize=True,
        )

    with pytest.raises(
            AssertionError,
            match="extra_fields and extra_fields_types must have the same keys",
    ):
        target_sync_test(
            TargetParquet(
                config=sample_config
                       | {
                           "extra_fields": "field1=value1,field2=1",
                           "extra_fields_types": "field2=integer",
                       }
            ),
            input=StringIO(example1_schema_messages["messages"]),
            finalize=True,
        )


def test_e2e_partition_cols_validation(
        monkeypatch, sample_config, example1_schema_messages
):
    """Test partition_cols validation"""
    with pytest.raises(AssertionError, match="partition_cols must be in the schema"):
        target_sync_test(
            TargetParquet(config=sample_config | {"partition_cols": "col_b"}),
            input=StringIO(example1_schema_messages["messages"]),
            finalize=True,
        )
def test_e2e_create_file_compression_method_default(
        monkeypatch, test_output_dir, sample_config, example1_schema_messages
):
    """Test that the target creates a file with the expected records"""
    monkeypatch.setattr("time.time", lambda: 1700000000)

    target_sync_test(
        TargetParquet(config=sample_config),
        input=StringIO(example1_schema_messages["messages"]),
        finalize=True,
    )

    assert (
            len(os.listdir(test_output_dir / example1_schema_messages["stream_name"])) == 1
    )

    files = os.listdir(test_output_dir / example1_schema_messages["stream_name"])
    assert files[0].endswith('.gz.parquet')


def test_e2e_create_file_compression_method_snappy(
        monkeypatch, test_output_dir, sample_config, example1_schema_messages
):
    """Test that the target creates a file with the expected records"""
    monkeypatch.setattr("time.time", lambda: 1700000000)

    target_sync_test(
        TargetParquet(
            config=sample_config | {"compression_method": "snappy"}),
        input=StringIO(example1_schema_messages["messages"]),
        finalize=True,
    )

    assert (
            len(os.listdir(test_output_dir / example1_schema_messages["stream_name"])) == 1
    )

    files = os.listdir(test_output_dir / example1_schema_messages["stream_name"])
    assert files[0].endswith('.snappy.parquet')
