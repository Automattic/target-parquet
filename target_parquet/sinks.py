"""parquet target sink class, which handles writing streams."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from singer_sdk.helpers._flattening import flatten_schema, flatten_record
from singer_sdk.sinks import BatchSink

from target_parquet.utils.parquet import (
    concat_tables,
    flatten_schema_to_pyarrow_schema,
    get_pyarrow_table_size,
    write_parquet_file,
)


class ParquetSink(BatchSink):
    """parquet target sink class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pyarrow_df = None
        self.destination_path = os.path.join(
            self.config.get("destination_path", "output"), self.stream_name
        )
        self.files_saved = 0
        self.flatten_max_level = self.config.get("max_flatten_level", 100)

        # Extra fields
        self.extra_values = (
            dict([kv.split("=") for kv in self.config["extra_fields"].split(",")])
            if self.config.get("extra_fields")
            else {}
        )
        self.extra_values_types = {}
        if self.config.get("extra_fields_types"):
            for field_type in self.config["extra_fields_types"].split(","):
                field_name, _type = field_type.split("=")
                self.extra_values_types[field_name] = {"type": [_type]}

        # Create pyarrow schema
        self.flatten_schema = flatten_schema(
            self.schema, max_level=self.flatten_max_level
        )
        self.flatten_schema.get("properties", {}).update(self.extra_values_types)
        self.pyarrow_schema = flatten_schema_to_pyarrow_schema(self.flatten_schema)

        self.partition_cols = (
            self.config["partition_cols"].split(",")
            if self.config.get("partition_cols")
            else None
        )

        self.validation()

    @property
    def basename_template(self) -> str:
        """Returns the basename template for the parquet file."""
        timestamp = datetime.fromtimestamp(
            self.sync_started_at / 1000, tz=timezone.utc
        ).strftime("%Y%m%d_%H%M%S")
        basename_template = f"{self.stream_name}-{timestamp}-{self.files_saved}-{{i}}"
        return basename_template

    def validation(self) -> None:
        """Extra fields and Partition Cols validation."""
        assert bool(self.extra_values) == bool(
            self.extra_values_types
        ), "extra_fields and extra_fields_types must be both set or both unset"
        if self.extra_values:
            assert (
                self.extra_values.keys() == self.extra_values_types.keys()
            ), "extra_fields and extra_fields_types must have the same keys"
        if self.partition_cols:
            assert set(self.partition_cols).issubset(
                set(self.pyarrow_schema.names)
            ), "partition_cols must be in the schema"

    @property
    def max_size(self) -> int:
        """Get max batch size.

        Returns:
            Max number of records to batch before `is_full=True`
        """
        return self.config.get("max_batch_size", 10000)

    def process_record(self, record: dict, context: dict) -> None:
        """Process the record.

        Args:
            record: Individual record in the stream.
            context: Stream partition or context dictionary.
        """
        record_flatten = (
            flatten_record(
                record,
                flattened_schema=self.flatten_schema,
                max_level=self.flatten_max_level,
            )
            | self.extra_values
        )
        super().process_record(record_flatten, context)

    def process_batch(self, context: dict) -> None:
        """Write out any prepped records and return once fully written.

        Args:
            context: Stream partition or context dictionary.
        """
        self.logger.info(
            f'Processing batch for {self.stream_name} with {len(context["records"])} records.'
        )
        self.pyarrow_df = concat_tables(
            context.get("records", []), self.pyarrow_df, self.pyarrow_schema
        )
        self.logger.info(
            f"Pyarrow table size: {self.pyarrow_df.nbytes} | ({len(self.pyarrow_df)} rows)"
        )
        del context["records"]

        self.write_file(new_file=get_pyarrow_table_size(self.pyarrow_df) > self.config["max_pyarrow_table_size"])

    def write_file(self, *, new_file: bool) -> None:
        """
        Write a local file.
        return the file path.

        :param new_file: bool - if True, write a new file.
        """
        if self.pyarrow_df is not None:
            write_parquet_file(
                self.pyarrow_df,
                self.destination_path,
                compression_method=self.config.get("compression_method", "gzip"),
                basename_template=self.basename_template,
                partition_cols=self.partition_cols,
            )
            if new_file:
                self.files_saved += 1
                self.pyarrow_df = None

    def clean_up(self) -> None:
        """Perform any clean up actions required at end of a stream."""
        self.write_file(new_file=True)
        super().clean_up()
