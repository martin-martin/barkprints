"""Create or manage barkprints user accounts from the command line.

There is no signup flow in the app — accounts are created here, e.g.::

    python -m barkprints.web.adduser alice
    python -m barkprints.web.adduser alice --password hunter2   # non-interactive
    python -m barkprints.web.adduser --list

The data directory is taken from ``BARKPRINTS_DATA_DIR`` (default ``data``), the
same location the web app uses, so accounts created here are immediately usable.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from .store import Store


def _data_dir() -> Path:
    return Path(os.environ.get("BARKPRINTS_DATA_DIR", "data")).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage barkprints user accounts.")
    parser.add_argument("username", nargs="?", help="Username to create or update.")
    parser.add_argument("--password", help="Password (omit to be prompted securely).")
    parser.add_argument(
        "--update", action="store_true",
        help="Update the password of an existing user instead of creating one.",
    )
    parser.add_argument("--list", action="store_true", help="List existing usernames and exit.")
    args = parser.parse_args(argv)

    store = Store(_data_dir())

    if args.list:
        users = store.list_users()
        if users:
            print("\n".join(users))
        else:
            print("(no users yet)")
        return 0

    if not args.username:
        parser.error("username is required (or use --list)")

    password = args.password
    if not password:
        password = getpass.getpass(f"Password for {args.username!r}: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.", file=sys.stderr)
            return 1

    try:
        if args.update:
            if not store.set_password(args.username, password):
                print(f"No such user: {args.username!r}", file=sys.stderr)
                return 1
            print(f"Updated password for {args.username!r}.")
        else:
            user = store.create_user(args.username, password)
            print(f"Created user {user.username!r} (id={user.id}).")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
