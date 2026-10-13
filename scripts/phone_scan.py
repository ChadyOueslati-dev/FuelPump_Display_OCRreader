"""Start the scan server so a phone on the same Wi-Fi can scan pump displays.

Usage:
    python scripts/phone_scan.py
    python scripts/phone_scan.py --access-code "a long secret"
    python scripts/phone_scan.py --http
    python scripts/phone_scan.py --tailscale --access-code "a long secret"
    python scripts/phone_scan.py --public --access-code "a much longer secret"
    python scripts/phone_scan.py --public      (with users from scripts/manage_users.py)

Open the printed address in the phone's browser, fit the pump display inside
the box and tap Scan. The result appears on the phone and in this terminal,
and every scan (picture + liters) is stored in data/. Ctrl+C stops it.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app  # noqa: E402
from app.ocr.engine import detect  # noqa: E402
from app.storage.db import ScanStore  # noqa: E402
from app.users import load_users  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MIN_PUBLIC_CODE_LENGTH = 16
USERS_FILE = DATA_DIR / "users.yaml"

# Inside data/, which is git-ignored: the private key must not be committed.
CERT_DIR = Path(__file__).resolve().parents[1] / "data" / "dev_cert"


def lan_address() -> str:
    """This PC's address on the local network."""
    # Connecting a UDP socket sends nothing; it only makes the OS pick the
    # network interface it would use, whose address we then read.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def tailscale_name() -> str | None:
    """This PC's name on the Tailscale network, e.g. laptop.tail1234.ts.net."""
    exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    try:
        out = subprocess.run(
            [exe, "status", "--json"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
        return json.loads(out)["Self"]["DNSName"].rstrip(".") or None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def dev_certificate(address: str) -> tuple[Path, Path]:
    """Self-signed certificate for this PC's address, created on first use.

    Phones only allow a web page to use the live camera over HTTPS. The
    certificate is for development on a local network only.
    """
    import ipaddress
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = CERT_DIR / f"{address}.cert.pem"
    key_path = CERT_DIR / f"{address}.key.pem"
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Pump OCR development")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(address)), x509.DNSName("localhost")]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="where scans are stored (default: data/)"
    )
    parser.add_argument("--access-code", help="one code shared by everyone (as the password)")
    parser.add_argument(
        "--users-file",
        type=Path,
        default=USERS_FILE,
        help="personal access codes, made with scripts/manage_users.py (default: data/users.yaml)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="plain HTTP: no certificate warning, but no live camera or framing box",
    )
    parser.add_argument(
        "--tailscale",
        action="store_true",
        help="reach the server from anywhere through Tailscale instead of the local Wi-Fi",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="like --tailscale, but for Tailscale Funnel: reachable from any phone on the internet",
    )
    parser.add_argument(
        "--allow-short-codes",
        action="store_true",
        help="with --public, accept personal codes shorter than 16 characters (5-digit PINs)",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="anyone can scan without signing in; the history still needs an access code",
    )
    args = parser.parse_args()

    args.tailscale = args.tailscale or args.public
    users = load_users(args.users_file)
    all_codes = list(users.values()) + ([args.access_code] if args.access_code else [])
    if args.tailscale and not all_codes:
        parser.error(
            "--tailscale and --public need access codes: create users with "
            "scripts/manage_users.py, or pass --access-code"
        )
    short_codes = min(len(code) for code in all_codes) < MIN_PUBLIC_CODE_LENGTH if all_codes else False
    if args.public and short_codes and not args.allow_short_codes:
        # On the public internet the codes are the only protection.
        parser.error(
            f"--public with access codes shorter than {MIN_PUBLIC_CODE_LENGTH} characters "
            "(such as 5-digit PINs) is easy to break into. Add --allow-short-codes to accept that risk."
        )
    if args.access_code and args.public and len(args.access_code) < MIN_PUBLIC_CODE_LENGTH:
        # A shared code has no name to go with it, so it must be long.
        parser.error(f"--public needs --access-code to have at least {MIN_PUBLIC_CODE_LENGTH} characters")

    address = lan_address()
    tls = {}
    if not args.http and not args.tailscale:
        cert_path, key_path = dev_certificate(address)
        tls = {"ssl_certfile": str(cert_path), "ssl_keyfile": str(key_path)}

    print("Loading the OCR models...")
    detect(np.full((64, 64, 3), 255, np.uint8))  # so the first scan is not slow

    print("\n" + "=" * 60)
    if args.tailscale:
        name = tailscale_name()
        if args.public:
            print("PUBLIC mode: any phone on the internet can open (no app needed):")
        else:
            print("On your phone (Tailscale app connected, any network), open:")
        print(f"\n    https://{name or '<this-pc>.<your-tailnet>.ts.net'}\n")
        print("Tailscale must be forwarding to this server. Run this once, in")
        print("another terminal (it stays on until 'tailscale funnel reset'):")
        command = "funnel" if args.public else "serve"
        print(f"\n    tailscale {command} --bg http://127.0.0.1:{args.port}\n")
    else:
        print("On your phone (same Wi-Fi as this PC), open:")
        print(f"\n    {'http' if args.http else 'https'}://{address}:{args.port}\n")
    if not args.http and not args.tailscale:
        print("The phone will warn that the connection is not private. That is")
        print("expected (the certificate is made by this PC). In Safari tap")
        print("'Show Details', then 'visit this website'. Then allow the camera.")
    if args.no_login:
        print("Scanning is OPEN: no sign-in. Anyone with the address can scan.")
        print("The history page (/scans) still asks for a user name and code.")
    elif any(len(code) < MIN_PUBLIC_CODE_LENGTH for code in users.values()):
        print("When asked to sign in: your own name as user name, your PIN as password.")
    elif all_codes:
        print("When asked to sign in: any user name, and the access code as password.")
    if users:
        print(f"Personal codes loaded for: {', '.join(sorted(users))}")
    print(f"Scans are stored in: {args.data_dir}  (history page: /scans)")
    print("Results appear below. Press Ctrl+C to stop.")
    print("=" * 60, flush=True)

    uvicorn.run(
        create_app(detect, ScanStore(args.data_dir), args.access_code, users, args.no_login),
        # With Tailscale only its own forwarder, on this PC, talks to the
        # server; nothing on the local network can reach it.
        host="127.0.0.1" if args.tailscale else "0.0.0.0",
        port=args.port,
        log_level="warning",
        **tls,
    )


if __name__ == "__main__":
    main()
