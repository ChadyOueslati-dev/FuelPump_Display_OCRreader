"""Per-person access codes, kept in a YAML file: name -> code.

The file holds the codes in plain text so they can be looked up and handed
out again. It lives in data/ (git-ignored). Anyone who can read the file can
use the server, so treat it like a password list.
"""

import secrets
from pathlib import Path

import yaml

# Letters and digits that are hard to confuse when typed on a phone
# (no 0/o, 1/l/i).
_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def generate_code() -> str:
    """A random code such as "k7m2p-x9qrt-4hwne-j3bvf" (about 99 bits)."""
    groups = ("".join(secrets.choice(_ALPHABET) for _ in range(5)) for _ in range(4))
    return "-".join(groups)


def generate_pin(taken: set[str] = frozenset()) -> str:
    """A random 5-digit PIN that is not in use and not an obvious pattern."""
    while True:
        pin = f"{secrets.randbelow(100000):05d}"
        obvious = len(set(pin)) == 1 or pin in "0123456789876543210"
        if pin not in taken and not obvious:
            return pin


def load_users(path: Path) -> dict[str, str]:
    """name -> code. An absent file means no users."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    users = {str(name): str(code) for name, code in (data.get("users") or {}).items()}
    if len(set(users.values())) != len(users):
        raise ValueError(f"{path}: two users have the same code")
    return users


def save_users(path: Path, users: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Access codes for the scan server: name -> code. Keep this file private.\n")
        yaml.safe_dump({"users": users}, f, sort_keys=True)
