# src/dcegm/cli.py
import argparse
import importlib.resources as pkg_resources
import shutil
from pathlib import Path


def copy_template(version: str, target_dir: Path):
    template_path = pkg_resources.files("dcegm.templates") / version

    if not template_path.exists():
        raise ValueError(f"Template version '{version}' does not exist.")

    for item in template_path.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def create_project(project_name: str, style: str):
    project_dir = Path(project_name).resolve()
    if project_dir.exists():
        print(f"Directory '{project_dir}' already exists.")
        return

    project_dir.mkdir(parents=True)
    copy_template(f"{style}", project_dir)
    print(f"Project '{project_name}' created with style {style}.")


def cli():
    parser = argparse.ArgumentParser(prog="dcegm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Initialize a new dcegm project")
    init.add_argument("project_name", type=str, help="Directory to create")
    init.add_argument(
        "--style",
        type=str,
        choices=["fullmodel", "simplemodel"],
        default="fullmodel",
        help="Project style to use (default: fullmodel)",
    )

    args = parser.parse_args()
    if args.command == "init":
        create_project(args.project_name, args.style)
