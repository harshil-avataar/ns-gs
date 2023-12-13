# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
import numpy as np
from PIL import Image


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML / or data path file.
    load_config: Optional[Path] = None  # overloaded based on dataset
    # Name of the output file
    output_dir: Path = Path("results/")
    #
    output_file: Path = Path("results.json")
    # Optional path to save rendered outputs to.
    # Optional checkpoint path for the same dataset
    checkpoint_path: Optional[Path] = None

    # def override_config(self):

    def main(self) -> None:
        """Main function."""
        # setup

        CONSOLE.log("[bold green] Processing NerfStudio Checkpoint")

        if self.checkpoint_path is not None:
            if self.checkpoint_path.exists():
                pass
            else:
                CONSOLE.log("[yellow] Checkpoint File is not valid, loading default checkpoint")
                self.checkpoint_path = None

        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config, self.checkpoint_path)
        assert self.output_file.suffix == ".json"

        self.render_output_path = Path(self.output_dir / "images/")
        if not self.render_output_path.exists():
            self.render_output_path.mkdir(parents=True)

        output_path = Path(self.output_dir / self.output_file)
        output = pipeline.get_average_eval_image_and_metrics_hb(output_path=self.render_output_path)
        # self.output_path.parent.mkdir(parents=True, exist_ok=True)

        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "config_path": str(self.load_config) if self.load_config is not None else "Generated",
            "checkpoint_path": str(checkpoint_path),
            "results": output[0],
            "results_all": output[1],
        }
        # Save output to output file
        output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
