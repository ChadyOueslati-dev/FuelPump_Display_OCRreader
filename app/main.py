"""Scan server: a phone scans a pump display, the server reads and stores it.

GET  /                  the scan page: live camera with a framing box
POST /scan              one photo in, liters out (accepted / review / retake); stored
GET  /scans             history: every stored scan with its picture and liters
GET  /scans/{id}/image  the picture of one scan
POST /scans/{id}/feedback  the person says whether the result was right

Every route needs the access code when one is configured. The fraud checks
from the spec are not built yet; POST /readings will add them.
"""

import html
import os
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.ocr.detection import Detection
from app.ocr.preprocess import decode_image
from app.pipeline import RETAKE, REVIEW, read_display
from app.report import format_result
from app.settings import load_validation_config
from app.storage.db import ScanStore

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# After this many wrong access codes within the window, every request is
# refused until the oldest of them is older than the window.
MAX_FAILED_LOGINS = 10
LOCKOUT_SECONDS = 300

# A personal code shorter than this (a 5-digit PIN, say) is easy to guess,
# so it only works together with its owner's name as the user name, and
# the lockout is stricter: 5 wrong tries per 15 minutes is at most 480
# guesses a day against 100,000 possible PINs per name.
LONG_CODE_LENGTH = 16
SHORT_CODE_MAX_FAILED_LOGINS = 5
SHORT_CODE_LOCKOUT_SECONDS = 900

PAGE = (Path(__file__).parent / "static" / "scan.html").read_text(encoding="utf-8")

HISTORY_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0c0e">
<link rel="icon" href="data:,">
<title>Scan history</title>
<style>
  :root {{ --line: rgba(255, 255, 255, 0.1); --muted: #a6acb8; --ok: #4ade80; --warn: #facc15; --bad: #fb923c; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0b0c0e; color: #f4f5f7; -webkit-font-smoothing: antialiased;
         font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 1; display: flex; align-items: center; gap: 12px;
           padding: calc(env(safe-area-inset-top, 0px) + 12px) 16px 12px;
           background: rgba(11, 12, 14, 0.8); border-bottom: 1px solid var(--line);
           -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px); }}
  header a {{ display: grid; place-items: center; width: 40px; height: 40px; border-radius: 50%;
             border: 1px solid var(--line); background: rgba(255, 255, 255, 0.06); }}
  header svg {{ width: 20px; height: 20px; fill: none; stroke: #f4f5f7; stroke-width: 1.8;
               stroke-linecap: round; stroke-linejoin: round; }}
  h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
  .count {{ margin-left: auto; color: var(--muted); font-size: 14px; }}
  main {{ display: grid; gap: 10px; padding: 12px 16px calc(env(safe-area-inset-bottom, 0px) + 24px); }}
  .scan {{ display: flex; gap: 14px; align-items: center; padding: 10px; border-radius: 18px;
          border: 1px solid var(--line); background: rgba(255, 255, 255, 0.04); }}
  .scan img {{ display: block; width: 84px; height: 84px; object-fit: cover; border-radius: 12px;
              background: #16181c; }}
  .scan > div {{ min-width: 0; }}
  .liters {{ font-size: 22px; font-weight: 700; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }}
  .accepted {{ color: var(--ok); }}
  .review {{ color: var(--warn); }}
  .retake {{ color: var(--bad); }}
  .meta {{ color: var(--muted); font-size: 13px; line-height: 1.45; overflow-wrap: anywhere; }}
  .meta:empty {{ display: none; }}
  .meta.accepted {{ color: var(--ok); }}
  .meta.retake {{ color: var(--bad); }}
  .empty {{ padding: 56px 16px; text-align: center; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <a href="/" aria-label="Back to scanning"><svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg></a>
  <h1>Scan history</h1>
  <span class="count">{count} scan(s)</span>
</header>
<main>
{rows}
</main>
<script>
// Stored times are UTC; show them in the phone's own time zone.
for (const el of document.querySelectorAll('[data-utc]')) {{
  el.textContent = new Date(el.dataset.utc).toLocaleString();
}}
</script>
</body>
</html>
"""

CORRECT, WRONG = "correct", "wrong"
# A liters value as a person types it: "22.93" or "22,93".
TYPED_LITERS = re.compile(r"\d{1,3}([.,]\d{1,3})?")


class Feedback(BaseModel):
    """What the person says about a scan."""

    correct: bool  # was the returned liters right?
    true_liters: str | None = None  # what the pump showed, if they typed it


HISTORY_ROW = """<div class="scan">
  <a href="/scans/{id}/image"><img src="/scans/{id}/image" loading="lazy" alt="scan picture"></a>
  <div>
    <div class="liters {status}">{summary}</div>
    <div class="meta">{user}</div>
    <div class="meta" data-utc="{created_at}">{created_at}</div>
    <div class="meta">{detail}</div>
    <div class="meta {feedback_class}">{feedback}</div>
  </div>
</div>"""


def create_app(
    detect: Callable[[np.ndarray], list[Detection]] | None = None,
    store: ScanStore | None = None,
    access_code: str | None = None,
    users: dict[str, str] | None = None,
    open_scanning: bool = False,
) -> FastAPI:
    """Build the app.

    detect:       the OCR function; defaults to PaddleOCR, tests pass a fake.
    store:        where scans are kept. None means nothing is stored.
    users:        name -> personal access code. Each scan is stored with the
                  name of the person whose code was used.
    access_code:  one code shared by everyone (stored as user "shared"). If
                  either is set, every request must supply a valid code as the password of
                  HTTP Basic authentication (any user name). The browser asks
                  for it once. None means open access: local development only.
    open_scanning: anyone may open the scan page and scan, without signing
                  in. The history and the stored pictures still need a code,
                  because they show what other people scanned.
    """
    if detect is None:
        from app.ocr.engine import detect

    basic = HTTPBasic(auto_error=False)
    failed_logins: deque[float] = deque()  # times of recent wrong access codes
    login_lock = threading.Lock()

    codes = dict(users or {})  # name -> code
    pin_users = {name for name, code in codes.items() if len(code) < LONG_CODE_LENGTH}
    if access_code is not None:
        codes["shared"] = access_code
    max_failed, lockout_seconds = (
        (SHORT_CODE_MAX_FAILED_LOGINS, SHORT_CODE_LOCKOUT_SECONDS)
        if pin_users
        else (MAX_FAILED_LOGINS, LOCKOUT_SECONDS)
    )

    def require_access(
        credentials: HTTPBasicCredentials | None = Depends(basic),
    ) -> str | None:
        """The name of the signed-in person, or None when access is open."""
        if not codes:
            return None
        now = time.monotonic()
        with login_lock:
            while failed_logins and now - failed_logins[0] > lockout_seconds:
                failed_logins.popleft()
            locked = len(failed_logins) >= max_failed
        # While locked, nobody gets in, not even with the right code: checking
        # the code at all would let a guesser keep guessing. The lockout is
        # shared by everyone because, behind a forwarder, all requests appear
        # to come from the same address.
        if locked:
            raise HTTPException(
                429,
                "Too many wrong access codes. Try again in a few minutes.",
                {"Retry-After": str(lockout_seconds)},
            )
        challenge = {"WWW-Authenticate": 'Basic realm="Pump scan"'}
        if credentials is None:  # the browser's first visit, not a guess
            raise HTTPException(401, "Access code required.", challenge)
        # A long code identifies the person by itself and the typed user
        # name is ignored. A short PIN only counts with its owner's name.
        # Every code is compared, without stopping at a match, so timing
        # does not reveal which one matched.
        supplied = credentials.password.encode()
        typed_name = credentials.username.strip().lower()
        matches = [
            name for name, code in codes.items()
            if secrets.compare_digest(supplied, code.encode())
            and (name not in pin_users or name.lower() == typed_name)
        ]
        if not matches:
            with login_lock:
                failed_logins.append(now)
            raise HTTPException(401, "Wrong access code.", challenge)
        return matches[0]

    def nobody() -> None:
        return None

    scan_access = nobody if open_scanning else require_access

    app = FastAPI(title="Pump display OCR")
    config = load_validation_config()
    ocr_lock = threading.Lock()  # the OCR engine is not safe to call concurrently

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(scan_access)])
    def page() -> str:
        return PAGE

    @app.post("/scan")
    def scan(photo: UploadFile, user: str | None = Depends(scan_access)) -> dict:
        data = photo.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Photo is too large.")
        image = decode_image(data)
        if image is None:
            raise HTTPException(400, "That file is not a readable image.")

        with ocr_lock:
            started = time.perf_counter()
            result = read_display(image, config, detect)
            seconds = round(time.perf_counter() - started, 2)

        # Only liters is returned; the other rows are used to check it. On a
        # "review" (display without a price row) liters is returned unverified.
        liters = str(result.liters.value) if result.liters else None
        confidence = result.liters.confidence if result.liters else None

        # Retakes are stored too: they are the photos the system needs to
        # get better at.
        scan_id = None
        if store is not None:
            scan_id = store.save(
                image, result.status, liters, confidence, result.reasons, seconds, user
            )

        height, width = image.shape[:2]
        print(f"\n== scan {scan_id or ''}  ({width}x{height})" + (f"  by {user}" if user else ""))
        print(format_result(result))
        print(f"Scan time: {seconds:.2f} s", flush=True)

        return {
            "id": scan_id,
            "status": result.status,
            "liters": liters,
            "confidence": confidence,
            "flags": result.reasons,
            # Time spent reading the photo on the server, without upload time.
            "scan_seconds": seconds,
        }

    @app.get("/scans", response_class=HTMLResponse, dependencies=[Depends(require_access)])
    def history() -> str:
        scans = store.recent() if store is not None else []
        rows = []
        for s in scans:
            if s["status"] == RETAKE:
                summary, detail = "Retake", ", ".join(s["flags"])
            else:
                summary = f"{s['liters']} L"
                detail = f"confidence {s['confidence']:.3f}"
                if s["status"] == REVIEW:
                    summary += " (not verified)"
                    detail += ", " + ", ".join(s["flags"])
            said = ""
            if s["feedback"] == CORRECT:
                said = "confirmed correct"
            elif s["feedback"] == WRONG:
                said = "marked wrong" + (f", pump showed {s['true_liters']} L" if s["true_liters"] else "")
            elif s["true_liters"]:
                said = f"pump showed {s['true_liters']} L"
            rows.append(
                HISTORY_ROW.format(
                    id=html.escape(s["id"]),
                    status=html.escape(s["status"]),
                    summary=html.escape(summary),
                    user=html.escape(f"by {s['user']}" if s["user"] else ""),
                    created_at=html.escape(s["created_at"]),
                    detail=html.escape(detail),
                    feedback=html.escape(said),
                    feedback_class="accepted" if s["feedback"] == CORRECT else "retake",
                )
            )
        empty = '<div class="empty">No scans yet.</div>'
        return HISTORY_PAGE.format(count=len(scans), rows="\n".join(rows) or empty)

    @app.post("/scans/{scan_id}/feedback")
    def give_feedback(
        scan_id: str, body: Feedback, user: str | None = Depends(scan_access)
    ) -> dict:
        """The person who scanned says whether the result was right.

        This is what turns everyday scans into labelled test data. For a
        retake there is no result to judge, only the true value to record.
        """
        scan = store.get(scan_id) if store is not None else None
        if scan is None or scan["user"] != user:
            raise HTTPException(404, "No such scan.")
        typed = (body.true_liters or "").strip()
        if typed and not TYPED_LITERS.fullmatch(typed):
            raise HTTPException(422, "Type the liters as a number, for example 22.93")
        typed = typed.replace(",", ".") or None

        if scan["liters"] is None:  # a retake
            if typed is None:
                raise HTTPException(422, "Type the liters the pump showed.")
            feedback, true_liters = None, typed
        elif body.correct:
            feedback, true_liters = CORRECT, scan["liters"]
        else:
            feedback, true_liters = WRONG, typed
        store.set_feedback(scan_id, feedback, true_liters)
        return {"id": scan_id, "feedback": feedback, "true_liters": true_liters}

    @app.get("/scans/{scan_id}/image", dependencies=[Depends(require_access)])
    def scan_image(scan_id: str) -> FileResponse:
        path = store.image_path(scan_id) if store is not None else None
        if path is None:
            raise HTTPException(404, "No such scan.")
        return FileResponse(path, media_type="image/jpeg")

    return app


def production_app() -> FastAPI:
    """The app for a hosted server, configured from environment variables.

    PUMP_OCR_ACCESS_CODE  required: the server refuses to start without it,
                          so it can never be on the internet unprotected.
    PUMP_OCR_DATA_DIR     where scans are stored (default: /data).
    """
    access_code = os.environ.get("PUMP_OCR_ACCESS_CODE", "")
    if len(access_code) < 12:
        raise RuntimeError(
            "Set PUMP_OCR_ACCESS_CODE to a secret of at least 12 characters before starting."
        )
    data_dir = Path(os.environ.get("PUMP_OCR_DATA_DIR", "/data"))
    return create_app(store=ScanStore(data_dir), access_code=access_code)
