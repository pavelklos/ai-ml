#!/usr/bin/env python
"""Virtual Environment Manager

A simple utility script to manage Python virtual environments.
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def list_environments(base_dir):
    """List all virtual environments in the specified directory."""
    base_path = Path(base_dir).expanduser()
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    print(f"Virtual environments in {base_dir}:")
    venvs = []

    # Look for directories that contain bin/activate or Scripts/activate
    for path in base_path.iterdir():
        if not path.is_dir():
            continue

        # Check for Unix-style activation script
        if (path / "bin" / "activate").exists():
            venvs.append(path.name)
        # Check for Windows-style activation script
        elif (path / "Scripts" / "activate").exists():
            venvs.append(path.name)

    if venvs:
        for i, venv in enumerate(sorted(venvs), 1):
            print(f"{i}. {venv}")
    else:
        print("No virtual environments found.")

def create_environment(base_dir, name, python_version=None):
    """Create a new virtual environment with the specified name."""
    base_path = Path(base_dir).expanduser()
    venv_path = base_path / name

    if venv_path.exists():
        print(f"Error: Environment '{name}' already exists.")
        return False

    if not base_path.exists():
        try:
            base_path.mkdir(parents=True)
            print(f"Created directory {base_dir}")
        except Exception as e:
            print(f"Error creating directory {base_dir}: {e}")
            return False

    # Determine Python executable
    if python_version:
        python_exe = f"python{python_version}"
    else:
        python_exe = "python"

    try:
        # Create the virtual environment
        subprocess.run([python_exe, "-m", "venv", str(venv_path)], check=True)
        print(f"Created virtual environment: {name} in {base_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def delete_environment(base_dir, name):
    """Delete the specified virtual environment."""
    base_path = Path(base_dir).expanduser()
    venv_path = base_path / name

    if not venv_path.exists():
        print(f"Error: Environment '{name}' does not exist.")
        return False

    try:
        shutil.rmtree(venv_path)
        print(f"Deleted virtual environment: {name}")
        return True
    except Exception as e:
        print(f"Error deleting environment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manage Python virtual environments.")

    # Default environment directory
    default_venv_dir = os.path.expanduser("~/.virtualenvs")

    parser.add_argument(
        "--dir", "-d",
        default=default_venv_dir,
        help=f"Base directory for virtual environments (default: {default_venv_dir})"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List virtual environments")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new virtual environment")
    create_parser.add_argument("name", help="Name of the virtual environment")
    create_parser.add_argument(
        "--python", "-p", 
        help="Python version to use (e.g., '3.9')"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a virtual environment")
    delete_parser.add_argument("name", help="Name of the virtual environment to delete")

    args = parser.parse_args()

    if args.command == "list":
        list_environments(args.dir)
    elif args.command == "create":
        create_environment(args.dir, args.name, args.python)
    elif args.command == "delete":
        delete_environment(args.dir, args.name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
