#!/usr/bin/env python3
"""Audit released-code held-out dataset cardinality before positioning."""

from __future__ import annotations

import argparse
import json

from held_out import (
    cardinality_record,
    common_input_arguments,
    inputs_from_args,
    prepare_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common_input_arguments(parser)
    args = parser.parse_args()
    spec, inputs = inputs_from_args(args)
    prepared = prepare_dataset(spec, inputs)
    print(json.dumps(cardinality_record(prepared), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
