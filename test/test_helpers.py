import pyarrow as pa
import logging

import pytest

from target_parquet.helpers import flatten, flatten_schema, flatten_schema_to_pyarrow_schema


def test_flatten():
    in_dict = {
        "key_1": 1,
        "key_2": {"key_3": 2, "key_4": {"key_5": 3, "key_6": ["10", "11"]}},
    }
    expected = {
        "key_1": 1,
        "key_2__key_3": 2,
        "key_2__key_4__key_5": 3,
        "key_2__key_4__key_6": '["10", "11"]',
    }

    flat_schema = {
        "key_1": "integer",
        "key_2__key_3": ["null", "integer"],
        "key_2__key_4__key_5": ["null", "integer"],
        "key_2__key_4__key_6": "string"
    }
    output = flatten(in_dict, flat_schema)
    assert output == expected


def test_flatten_with_empty_object():
    in_dict = {
        "key_1": 1,
        "key_2": None,
    }
    expected = {
        "key_1": 1,
        "key_2__key_3": None,
        "key_2__key_4__key_5": None,
        "key_2__key_4__key_6": None,
    }

    flat_schema = {
        "key_1": "integer",
        "key_2__key_3": ["null", "integer"],
        "key_2__key_4__key_5": ["null", "integer"],
        "key_2__key_4__key_6": "string"
    }
    output = flatten(in_dict, flat_schema)
    assert output == expected


def test_flatten_schema():
    in_dict = {
        "key_1": {"type": ["null", "integer"]},
        "key_2": {
            "type": ["null", "object"],
            "properties": {
                "key_3": {"type": ["null", "string"]},
                "key_4": {
                    "type": ["null", "object"],
                    "properties": {
                        "key_5": {"type": ["null", "integer"]},
                        "key_6": {
                            "type": ["null", "array"],
                            "items": {
                                "type": ["null", "object"],
                                "properties": {
                                    "key_7": {"type": ["null", "number"]},
                                    "key_8": {"type": ["null", "string"]},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    expected = {
        "key_1": ["null", "integer"],
        "key_2__key_3": ["null", "string"],
        "key_2__key_4__key_5": ["null", "integer"],
        "key_2__key_4__key_6": ["null", "array"]
    }

    output = flatten_schema(in_dict)
    assert output == expected


def test_flatten_schema_2(caplog):
    in_dict = {
        "id": {"type": "integer"},
        "created_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
        "email": {"type": "string"},
        "last_surveyed": {
            "anyOf": [{"type": "null"}, {"type": "string", "format": "date-time"}]
        },
        "external_created_at": {"type": ["integer", "null"]},
        "page_views_count": {"type": "integer"},
    }

    expected = {
        "id": "integer",
        "created_at": "string",
        "updated_at": "string",
        "email": "string",
        "last_surveyed": None,
        "external_created_at": ["integer", "null"],
        "page_views_count": "integer",
    }

    with caplog.at_level(logging.WARNING):
        output = flatten_schema(in_dict)
        for record in caplog.records:
            assert (
                "SCHEMA with limited support on field last_surveyed" in record.message
            )
    assert output == expected


def test_flatten_schema_empty():
    in_dict = dict()
    assert dict() == flatten_schema(in_dict)


def test_flatten_schema_to_pyarrow_schema():
    in_dict = {
        "id": "integer",
        "created_at": "string",
        "updated_at": "string",
        "email": "string",
        "email_list": ["array", "null"],
        "external_created_at": ["integer", "null"],
        "page_views_count": "integer",
        "only_null_datatype": ["null"],
        "page_views_avg": ["number", "null"],
    }

    expected = pa.schema([
        pa.field("id", pa.int64(), False),
        pa.field("created_at", pa.string(), False),
        pa.field("updated_at", pa.string(), False),
        pa.field("email", pa.string(), False),
        pa.field("email_list", pa.string(), True),
        pa.field("external_created_at", pa.int64(), True),
        pa.field("page_views_count", pa.int64(), False),
        pa.field("page_views_avg", pa.float64(), True),
        pa.field("only_null_datatype", pa.string(), True),
    ])

    assert expected.equals(flatten_schema_to_pyarrow_schema(in_dict, [
        "id",
        "created_at",
        "updated_at",
        "email",
        "email_list",
        "external_created_at",
        "page_views_count",
        "page_views_avg",
        "only_null_datatype"
    ]))


def test_flatten_schema_to_pyarrow_schema_type_not_defined():
    in_dict = {
        "created_at": "new-type",
    }

    with pytest.raises(NotImplementedError):
        flatten_schema_to_pyarrow_schema(in_dict, ["created_at"])


def test_flatten_array_fields():
    in_dict = {
        "int_array": [1, 2, 3],
        "array_of_int_array": [1, 2, [3, 4]],
        "array_of_mixed_types": ["a", {"b":"value"}, ["c", "d"]],
        "string_array": ["a", "b", "c"],
        "object_array": [{"int": 1}, {"string1": "aaa'aaa"}, {"string2": 'aaa"aaa'}, {"array": [1, 2, 3]}, {'true': True}, {'false': False}, {'null': None}],
    }
    expected = {
        'int_array': '[1, 2, 3]',
        'array_of_int_array': '[1, 2, [3, 4]]',
        "array_of_mixed_types": '["a", {"b": "value"}, ["c", "d"]]',
        'string_array': '["a", "b", "c"]',
        'object_array': '[{"int": 1}, {"string1": "aaa\'aaa"}, {"string2": "aaa\\"aaa"}, {"array": [1, 2, 3]}, {"true": true}, {"false": false}, {"null": null}]'
    }

    flat_schema = {
        "int_array": "array",
        "array_of_int_array": "array",
        "array_of_mixed_types": "array",
        "string_array": ["null", "array"],
        "object_array": "array",
    }
    output = flatten(in_dict, flat_schema)
    assert output == expected
