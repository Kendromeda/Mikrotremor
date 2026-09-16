"""Command line entry point.

Commands are thin: they parse arguments, call the library, and print. Any
failure surfaces as a non-zero exit code with the failure list attached, so a
broken build cannot be mistaken for a quiet one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mhvsr_vs30 import __version__
from mhvsr_vs30.contracts import Recording
from mhvsr_vs30.exceptions import (
    ConfigError,
    ManifestValidationError,
    MhvsrVs30Error,
    RecordingReadError,
)
from mhvsr_vs30.io.registry import get_reader
from mhvsr_vs30.io.resolve import resolve_request
from mhvsr_vs30.manifest.builder import build_manifest
from mhvsr_vs30.manifest.schemas import Manifest, load_source_config
from mhvsr_vs30.manifest.store import read_manifest, write_manifest
from mhvsr_vs30.manifest.validation import validate_manifest
from mhvsr_vs30.preprocessing.batch import run_profile, write_report
from mhvsr_vs30.preprocessing.compare import compare_profiles
from mhvsr_vs30.preprocessing.config import load_profile

__all__ = ["main"]

DEFAULT_CACHE = Path("artifacts/cache/checksums.json")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_manifest_build(args: argparse.Namespace) -> int:
    config = load_source_config(args.config)
    manifest = build_manifest(config, args.workspace, cache_path=args.cache)
    output = write_manifest(manifest, args.output, config)
    summary = manifest.summary()
    summary["output_dir"] = str(output)
    _emit(summary)
    return 0


def _cmd_manifest_validate(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest_dir)
    config = load_source_config(args.config)
    failures = validate_manifest(manifest, config)
    _emit(
        {
            "manifest_dir": str(args.manifest_dir),
            "source_id": manifest.source.source_id,
            "failure_count": len(failures),
            "failures": failures,
            "status": "passed" if not failures else "failed",
        }
    )
    return 0 if not failures else 1


def _find_recording(manifest: Manifest, recording_id: str) -> Recording:
    """Accept a unique id prefix, so an operator does not paste 64 hex characters."""
    matches = [r for r in manifest.recordings if r.recording_id.startswith(recording_id)]
    if not matches:
        raise RecordingReadError(f"no recording in this manifest starts with {recording_id!r}")
    if len(matches) > 1:
        raise RecordingReadError(
            f"{recording_id!r} matches {len(matches)} recordings; use a longer prefix"
        )
    return matches[0]


def _cmd_recording_inspect(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    config = load_source_config(args.config)
    recording = _find_recording(manifest, args.recording_id)
    request = resolve_request(manifest, recording, config, args.workspace)

    result = get_reader(recording.raw_format).read(request)
    payload = {"reader": recording.raw_format, "relative_path": request.asset.relative_path}
    payload.update(result.structural_qc())
    _emit(payload)
    return 0


def _cmd_recording_validate_all(args: argparse.Namespace) -> int:
    """Read every recording and report structural QC, hiding no failure."""
    manifest = read_manifest(args.manifest)
    config = load_source_config(args.config)

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for recording in manifest.recordings:
        try:
            request = resolve_request(manifest, recording, config, args.workspace)
            result = get_reader(recording.raw_format).read(request)
        except MhvsrVs30Error as error:
            failed.append(
                {
                    "recording_id": recording.recording_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            entry = {"relative_path": request.asset.relative_path}
            entry.update(result.structural_qc())
            passed.append(entry)

    report = {
        "source_id": manifest.source.source_id,
        "manifest_dir": str(args.manifest),
        "recording_count": len(manifest.recordings),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "status": "passed" if not failed else "failed",
        "passed": passed,
        "failed": failed,
    }
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _emit({key: value for key, value in report.items() if key != "passed"})
    return 0 if not failed else 1


def _cmd_preprocess_run(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    source = load_source_config(args.config)
    profile = load_profile(args.profile)

    snapshot_path = Path(args.manifest) / "snapshot.json"
    snapshot_hash = None
    if snapshot_path.is_file():
        snapshot_hash = json.loads(snapshot_path.read_text(encoding="utf-8")).get("config_hash")

    summary = run_profile(
        manifest,
        source,
        profile,
        args.workspace,
        curves_root=args.curves,
        index_root=args.indexes,
        snapshot_hash=snapshot_hash,
    )
    if args.report is not None:
        write_report(summary, args.report)
    _emit(summary)
    return 0 if not summary["failure_reasons"] else 1


def _cmd_preprocess_compare(args: argparse.Namespace) -> int:
    report = compare_profiles(args.source_id, args.left, args.right, args.indexes, args.curves)
    if args.report is not None:
        write_report(report, args.report)
    _emit(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mhvsr-vs30", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="group", required=True)

    manifest = subcommands.add_parser("manifest", help="build and check dataset manifests")
    manifest_actions = manifest.add_subparsers(dest="action", required=True)

    build = manifest_actions.add_parser("build", help="build a manifest from a dataset config")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--workspace", default=Path("."), type=Path)
    build.add_argument("--cache", default=DEFAULT_CACHE, type=Path)
    build.set_defaults(handler=_cmd_manifest_build)

    validate = manifest_actions.add_parser("validate", help="re-check a manifest on disk")
    validate.add_argument("manifest_dir", type=Path)
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=_cmd_manifest_validate)

    recording = subcommands.add_parser("recording", help="read and check raw recordings")
    recording_actions = recording.add_subparsers(dest="action", required=True)

    inspect = recording_actions.add_parser("inspect", help="parse one recording and report QC")
    inspect.add_argument("--manifest", required=True, type=Path)
    inspect.add_argument("--config", required=True, type=Path)
    inspect.add_argument("--recording-id", required=True)
    inspect.add_argument("--workspace", default=Path("."), type=Path)
    inspect.set_defaults(handler=_cmd_recording_inspect)

    validate_all = recording_actions.add_parser(
        "validate-all", help="parse every recording in a manifest"
    )
    validate_all.add_argument("--manifest", required=True, type=Path)
    validate_all.add_argument("--config", required=True, type=Path)
    validate_all.add_argument("--workspace", default=Path("."), type=Path)
    validate_all.add_argument("--report", default=None, type=Path)
    validate_all.set_defaults(handler=_cmd_recording_validate_all)

    preprocess = subcommands.add_parser("preprocess", help="turn recordings into mHVSR curves")
    preprocess_actions = preprocess.add_subparsers(dest="action", required=True)

    run = preprocess_actions.add_parser("run", help="process every recording under a profile")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--profile", required=True, type=Path)
    run.add_argument("--workspace", default=Path("."), type=Path)
    run.add_argument("--curves", default=Path("artifacts/curves"), type=Path)
    run.add_argument("--indexes", default=Path("artifacts/curve_indexes"), type=Path)
    run.add_argument("--report", default=None, type=Path)
    run.set_defaults(handler=_cmd_preprocess_run)

    compare = preprocess_actions.add_parser("compare", help="compare two profiles")
    compare.add_argument("--source-id", required=True, dest="source_id")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--curves", default=Path("artifacts/curves"), type=Path)
    compare.add_argument("--indexes", default=Path("artifacts/curve_indexes"), type=Path)
    compare.add_argument("--report", default=None, type=Path)
    compare.set_defaults(handler=_cmd_preprocess_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, ManifestValidationError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except MhvsrVs30Error as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
