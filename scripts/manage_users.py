"""Create, list and remove the people who may use the scan server.

Usage:
    python scripts/manage_users.py add sara ahmed      (prints their new codes)
    python scripts/manage_users.py list                (names only)
    python scripts/manage_users.py list --show-codes
    python scripts/manage_users.py rename user1 sara
    python scripts/manage_users.py remove ahmed
    python scripts/manage_users.py new-code sara       (replaces a lost or leaked code)
    python scripts/manage_users.py add omar --pin      (a 5-digit PIN instead)
    python scripts/manage_users.py new-code sara --pin

With a long code, a person signs in with any user name and the code as the
password. With a 5-digit PIN, the user name must be their own name.
Restart the scan server after any change.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.users import generate_code, generate_pin, load_users, save_users  # noqa: E402

USERS_FILE = Path(__file__).resolve().parents[1] / "data" / "users.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=USERS_FILE)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("names", nargs="+")
    add.add_argument("--quiet", action="store_true", help="do not print the new codes")
    add.add_argument("--pin", action="store_true", help="a 5-digit PIN instead of a long code")
    show = commands.add_parser("list")
    show.add_argument("--show-codes", action="store_true")
    rename = commands.add_parser("rename")
    rename.add_argument("old")
    rename.add_argument("new")
    commands.add_parser("remove").add_argument("names", nargs="+")
    new_code = commands.add_parser("new-code")
    new_code.add_argument("names", nargs="+")
    new_code.add_argument("--quiet", action="store_true", help="do not print the new codes")
    new_code.add_argument("--pin", action="store_true", help="a 5-digit PIN instead of a long code")
    args = parser.parse_args()

    users = load_users(args.file)

    def new_code() -> str:
        return generate_pin(set(users.values())) if args.pin else generate_code()

    if args.command == "list":
        print(f"{len(users)} user(s) in {args.file}")
        for name, code in sorted(users.items()):
            print(f"  {name:<16} {code if args.show_codes else ''}".rstrip())
        return 0

    if args.command == "add":
        existing = [n for n in args.names if n in users]
        if existing:
            print(f"error: already exists: {', '.join(existing)}", file=sys.stderr)
            return 1
        for name in args.names:
            users[name] = new_code()
            print(f"added {name}" + ("" if args.quiet else f"   code: {users[name]}"))
    elif args.command == "rename":
        if args.old not in users or args.new in users:
            print("error: the old name must exist and the new name must be free", file=sys.stderr)
            return 1
        users[args.new] = users.pop(args.old)
        print(f"renamed {args.old} to {args.new} (same code)")
    else:
        unknown = [n for n in args.names if n not in users]
        if unknown:
            print(f"error: no such user: {', '.join(unknown)}", file=sys.stderr)
            return 1
        for name in args.names:
            if args.command == "remove":
                del users[name]
                print(f"removed {name}")
            else:
                users[name] = new_code()
                print(f"new code for {name}" + ("" if args.quiet else f": {users[name]}"))

    save_users(args.file, users)
    print("Restart the scan server for this to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
