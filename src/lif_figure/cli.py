"""Command-line interface for lif-figure."""

import re
import sys
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt

from lif_figure.config import load_config, DEFAULT_AUTO_CONTRAST_PERCENTILES
from lif_figure.reader import list_series, read_series, extract_series_metadata, LifFile
from lif_figure.figure import build_figure, build_rows_figure, compute_normalization_ranges


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as filename."""
    return re.sub(r'[/\\:*?"<>|]', "_", name)


def save_figure(fig, output_file: Path, background: str) -> None:
    """Save figure with proper background handling."""
    is_transparent = background.lower() == "transparent"
    facecolor = "none" if is_transparent else background
    fig.savefig(
        output_file,
        format="pdf",
        bbox_inches="tight",
        facecolor=facecolor,
        transparent=is_transparent,
    )


def parse_series_indices(spec: str, max_index: int) -> list[int]:
    """Parse index specification like '0-2,5,8-10' into [0,1,2,5,8,9,10]."""
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))

    # Validate all in range
    for i in indices:
        if i < 0 or i >= max_index:
            raise ValueError(f"Index {i} out of range (0-{max_index - 1})")

    return sorted(set(indices))  # Dedupe and sort


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
    "--series-index", "-si",
    default=None,
    help="Series indices to process (0-indexed), comma-separated with ranges (e.g., '0-2,5,8-10')",
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
@click.option(
    "--auto-contrast", "-a",
    "auto_contrast",
    default=None,
    is_flag=False,
    flag_value="default",
    help="Enable auto-contrast (optional: LOW,HIGH percentiles, e.g., '0.5,99.5')",
)
@click.option(
    "--no-metadata",
    is_flag=True,
    default=False,
    help="Hide acquisition metadata table from output",
)
@click.option(
    "--per-slice-norm",
    is_flag=True,
    default=False,
    help="Normalize each Z-slice independently (default: normalize across stack)",
)
@click.option(
    "--background", "-bg",
    default=None,
    help="Background color: black, white, transparent, or any color name/hex (default: black)",
)
def main(
    input_file: str,
    channels: str,
    series: Optional[str],
    series_index: Optional[str],
    output: str,
    zstack: str,
    config: Optional[str],
    auto_contrast: Optional[str],
    no_metadata: bool,
    per_slice_norm: bool,
    background: Optional[str],
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

    # Validate and normalize zstack mode
    valid_modes = ["max", "frames", "rows"]
    mode_base = zstack.split(":")[0]  # Handle max:5-15 syntax
    if mode_base == "row":
        zstack = "rows"  # Accept "row" as alias
        mode_base = "rows"
    if mode_base not in valid_modes:
        click.echo(f"Error: Invalid zstack mode: '{zstack}'", err=True)
        click.echo(f"Valid modes: {', '.join(valid_modes)}, max:START-END", err=True)
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

    # Handle auto-contrast flag
    if auto_contrast is not None:
        if auto_contrast == "default":
            # Use config file value or defaults
            if cfg.auto_contrast_percentiles is None:
                cfg.auto_contrast_percentiles = DEFAULT_AUTO_CONTRAST_PERCENTILES
        else:
            # Parse custom percentiles
            try:
                parts = auto_contrast.split(",")
                if len(parts) != 2:
                    raise ValueError("Expected two values")
                cfg.auto_contrast_percentiles = (float(parts[0]), float(parts[1]))
            except ValueError:
                click.echo(f"Error: Invalid auto-contrast format: '{auto_contrast}'", err=True)
                click.echo("Expected format: LOW,HIGH (e.g., '0.5,99.5')", err=True)
                sys.exit(1)

    # Handle background color override
    if background is not None:
        cfg.background = background

    # Get series to process
    all_series = list_series(input_path)

    # Check mutual exclusivity
    if series and series_index:
        click.echo("Error: Cannot use both --series and --series-index. Use one or the other.", err=True)
        sys.exit(1)

    if series_index:
        try:
            indices = parse_series_indices(series_index, len(all_series))
            series_names = [all_series[i] for i in indices]
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            click.echo(f"File contains {len(all_series)} series (indices 0-{len(all_series) - 1}).", err=True)
            sys.exit(1)
    elif series:
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

    # Open LIF file for metadata extraction
    lif = LifFile(str(input_path))
    show_metadata = not no_metadata

    # Process each series
    for i, series_name in enumerate(series_names, 1):
        click.echo(f"  Series {i} of {len(series_names)}: \"{series_name}\"", nl=False)

        try:
            data, pixel_size_um, z_pixel_size_um = read_series(input_path, series_name, zstack)

            # Extract metadata if needed
            metadata = None
            if show_metadata:
                metadata = extract_series_metadata(lif, series_name)

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
                fig = build_figure(
                    data, channel_names, cfg, pixel_size_um,
                    metadata=metadata, show_metadata=show_metadata
                )

                safe_name = sanitize_filename(series_name)
                output_file = output_path / f"{safe_name}.pdf"
                save_figure(fig, output_file, cfg.background)
                plt.close(fig)

                click.echo(f" → {output_file}")

            elif zstack == "frames":
                # Separate PDF per Z frame
                subdir = output_path / sanitize_filename(series_name)
                subdir.mkdir(exist_ok=True)

                # Compute stack-wide normalization ranges (unless per-slice)
                norm_ranges = None
                if not per_slice_norm:
                    norm_ranges = compute_normalization_ranges(
                        data, cfg.auto_contrast_percentiles
                    )

                for z in range(data.shape[0]):
                    fig = build_figure(
                        data[z], channel_names, cfg, pixel_size_um,
                        metadata=metadata, show_metadata=show_metadata,
                        normalization_ranges=norm_ranges,
                    )
                    output_file = subdir / f"z{z:02d}.pdf"
                    save_figure(fig, output_file, cfg.background)
                    plt.close(fig)

                click.echo(f" → {subdir}/ ({data.shape[0]} frames)")

            elif zstack == "rows":
                # All Z frames as rows in single PDF
                norm_ranges = None
                if not per_slice_norm:
                    norm_ranges = compute_normalization_ranges(
                        data, cfg.auto_contrast_percentiles
                    )

                fig = build_rows_figure(
                    data, channel_names, cfg, pixel_size_um,
                    z_pixel_size_um=z_pixel_size_um,
                    metadata=metadata, show_metadata=show_metadata,
                    normalization_ranges=norm_ranges,
                )

                safe_name = sanitize_filename(series_name)
                output_file = output_path / f"{safe_name}.pdf"
                save_figure(fig, output_file, cfg.background)
                plt.close(fig)

                click.echo(f" → {output_file} ({data.shape[0]} rows)")

        except Exception as e:
            click.echo(f" [ERROR: {e}]")
            continue

    click.echo(f"Done. Figures saved to {output_path}/")


if __name__ == "__main__":
    main()
