"""Command-line interface for lif-figure."""

import re
import sys
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt

from lif_figure.config import load_config
from lif_figure.reader import list_series, read_series
from lif_figure.figure import build_figure


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as filename."""
    return re.sub(r'[/\\:*?"<>|]', "_", name)


@click.command()
@click.argument("input_file", type=click.Path(exists=False))
@click.option(
    "--channels", "-c",
    required=True,
    help="Channel names, comma-separated (e.g., 'DAPI,GFP,mCherry')",
)
@click.option(
    "--series", "-s",
    default=None,
    help="Series to process, comma-separated (default: all)",
)
@click.option(
    "--output", "-o",
    default="./figures",
    type=click.Path(),
    help="Output directory (default: ./figures)",
)
@click.option(
    "--zstack", "-z",
    default="max",
    help="Z-stack mode: max, max:START-END, frames, rows (default: max)",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Optional YAML config file",
)
def main(
    input_file: str,
    channels: str,
    series: Optional[str],
    output: str,
    zstack: str,
    config: Optional[str],
) -> None:
    """Generate publication-ready figure panels from LIF files.

    Each series produces a PDF with grayscale channel panels and a color merge.
    """
    input_path = Path(input_file)
    output_path = Path(output)
    config_path = Path(config) if config else None

    # Validate input file
    if not input_path.exists():
        click.echo(f"Error: File not found: {input_file}", err=True)
        sys.exit(1)

    if not input_path.suffix.lower() == ".lif":
        click.echo(f"Error: Not a LIF file: {input_file}", err=True)
        sys.exit(1)

    # Parse channel names
    channel_names = [c.strip() for c in channels.split(",")]

    # Load config
    cfg = load_config(config_path)

    # Auto-detect config in current directory
    if config_path is None:
        auto_config = Path("lif-figure.yaml")
        if auto_config.exists():
            cfg = load_config(auto_config)

    # Get series to process
    all_series = list_series(input_path)

    if series:
        series_names = [s.strip() for s in series.split(",")]
        # Validate series names
        for name in series_names:
            if name not in all_series:
                click.echo(f"Error: Series '{name}' not found in {input_file}", err=True)
                click.echo(f"Available series: {', '.join(all_series)}", err=True)
                sys.exit(1)
    else:
        series_names = all_series

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Processing {input_path.name}")

    # Process each series
    for i, series_name in enumerate(series_names, 1):
        click.echo(f"  Series {i} of {len(series_names)}: \"{series_name}\"", nl=False)

        try:
            data, pixel_size_um = read_series(input_path, series_name, zstack)

            # Validate channel count
            if data.ndim == 3:  # (C, H, W)
                n_channels = data.shape[0]
            else:  # (Z, C, H, W) for frames/rows mode
                n_channels = data.shape[1]

            if n_channels != len(channel_names):
                click.echo(f" [SKIP: {n_channels} channels, expected {len(channel_names)}]")
                continue

            # Handle different Z-stack modes
            if data.ndim == 3:
                # Single figure (max projection)
                fig = build_figure(data, channel_names, cfg, pixel_size_um)

                safe_name = sanitize_filename(series_name)
                output_file = output_path / f"{safe_name}.pdf"
                fig.savefig(output_file, format="pdf", bbox_inches="tight", facecolor=cfg.background)
                plt.close(fig)

                click.echo(f" → {output_file}")

            elif zstack == "frames":
                # Separate PDF per Z frame
                subdir = output_path / sanitize_filename(series_name)
                subdir.mkdir(exist_ok=True)

                for z in range(data.shape[0]):
                    fig = build_figure(data[z], channel_names, cfg, pixel_size_um)
                    output_file = subdir / f"z{z:02d}.pdf"
                    fig.savefig(output_file, format="pdf", bbox_inches="tight", facecolor=cfg.background)
                    plt.close(fig)

                click.echo(f" → {subdir}/ ({data.shape[0]} frames)")

            elif zstack == "rows":
                # All Z frames as rows in single PDF
                # TODO: Implement multi-row figure
                click.echo(" [SKIP: rows mode not yet implemented]")
                continue

        except Exception as e:
            click.echo(f" [ERROR: {e}]")
            continue

    click.echo(f"Done. Figures saved to {output_path}/")


if __name__ == "__main__":
    main()
