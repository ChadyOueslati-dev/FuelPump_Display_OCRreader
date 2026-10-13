import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app, production_app
from app.ocr.detection import Detection
from app.ocr.preprocess import MAX_SIDE, decode_image
from app.storage.db import ScanStore
from app.users import generate_code, generate_pin, load_users, save_users


def column(*texts):
    return [Detection(t, 0.99, (80, i * 50, 160, i * 50 + 40)) for i, t in enumerate(texts)]


def jpeg(width=400, height=300):
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), 90, np.uint8))
    assert ok
    return encoded.tobytes()


def client(*texts, **kwargs):
    return TestClient(create_app(detect=lambda image: column(*texts), **kwargs))


def post(c, data=None):
    return c.post("/scan", files={"photo": ("photo.jpg", data or jpeg(), "image/jpeg")})


def test_page_has_live_camera_with_roi_and_a_photo_fallback():
    response = client("50.00", "22.93", "2.18").get("/")
    assert response.status_code == 200
    assert "getUserMedia" in response.text and 'id="roi"' in response.text
    assert 'capture="environment"' in response.text
    assert 'id="zoom"' in response.text and "applyConstraints" in response.text


def test_accepted_scan(capsys):
    response = post(client("50.00", "22.93", "2.18"))
    assert response.status_code == 200
    body = response.json()
    assert body.pop("scan_seconds") >= 0
    assert body == {
        "id": None,
        "status": "accepted",
        "liters": "22.93",
        "confidence": 0.99,
        "flags": [],
    }
    printed = capsys.readouterr().out
    assert "Status: accepted" in printed and "22.93" in printed
    assert "Scan time:" in printed


def test_retake_returns_no_values():
    body = post(client("50.00", "22.93", "888.18")).json()
    assert body["status"] == "retake"
    assert body["liters"] is None
    assert body["confidence"] is None
    assert "amount_mismatch" in body["flags"]


def test_non_image_upload_is_rejected():
    response = post(client("50.00", "22.93", "2.18"), b"this is not a photo")
    assert response.status_code == 400


def test_nothing_is_stored_without_a_store():
    c = client("50.00", "22.93", "2.18")
    assert post(c).json()["id"] is None
    assert "0 scan(s)" in c.get("/scans").text


def test_scan_is_stored_with_its_picture_and_liters(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store)
    scan_id = post(c).json()["id"]

    (saved,) = store.recent()
    assert saved["id"] == scan_id
    assert (saved["status"], saved["liters"], saved["flags"]) == ("accepted", "22.93", [])
    assert (tmp_path / "scans" / saved["image_file"]).is_file()

    picture = c.get(f"/scans/{scan_id}/image")
    assert picture.status_code == 200
    assert picture.headers["content-type"] == "image/jpeg"
    assert decode_image(picture.content).shape == (300, 400, 3)

    page = c.get("/scans").text
    assert "1 scan(s)" in page and "22.93 L" in page and scan_id in page


def test_picture_and_record_are_stored_together_or_not_at_all(tmp_path, monkeypatch):
    store = ScanStore(tmp_path)
    post(client("50.00", "22.93", "2.18", store=store))
    assert store.check() == ([], [])

    # If the record cannot be written, no picture is left behind.
    def broken():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_connect", broken)
    with pytest.raises(OSError):
        store.save(np.full((10, 10, 3), 90, np.uint8), "accepted", "1.00", 0.99, [], 0.1)
    monkeypatch.undo()
    assert len(list((tmp_path / "scans").glob("*.jpg"))) == 1
    assert store.check() == ([], [])


def test_deleting_a_scan_removes_its_record_and_its_picture(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store)
    keep, remove = post(c).json()["id"], post(c).json()["id"]

    assert store.delete(remove) is True
    assert store.delete("no-such-scan") is False
    assert [s["id"] for s in store.recent()] == [keep]
    assert [p.stem for p in (tmp_path / "scans").glob("*.jpg")] == [keep]
    assert store.check() == ([], [])


def test_deleting_works_when_the_picture_is_already_gone(tmp_path):
    store = ScanStore(tmp_path)
    scan_id = post(client("50.00", "22.93", "2.18", store=store)).json()["id"]
    (tmp_path / "scans" / f"{scan_id}.jpg").unlink()
    assert store.delete(scan_id) is True
    assert store.recent() == [] and store.check() == ([], [])


def test_check_reports_pictures_and_records_that_do_not_match(tmp_path):
    store = ScanStore(tmp_path)
    post(client("50.00", "22.93", "2.18", store=store))
    (saved,) = store.recent()
    (tmp_path / "scans" / "stray.jpg").write_bytes(b"x")
    (tmp_path / "scans" / saved["image_file"]).unlink()
    assert store.check() == ([saved["image_file"]], ["stray.jpg"])


def test_retakes_are_stored_without_liters(tmp_path):
    store = ScanStore(tmp_path)
    post(client("50.00", "22.93", "888.18", store=store))
    (saved,) = store.recent()
    assert saved["status"] == "retake"
    assert saved["liters"] is None
    assert "amount_mismatch" in saved["flags"]


def test_scans_survive_a_restart(tmp_path):
    post(client("50.00", "22.93", "2.18", store=ScanStore(tmp_path)))
    post(client("21.80", "10.00", "2.18", store=ScanStore(tmp_path)))
    assert {s["liters"] for s in ScanStore(tmp_path).recent()} == {"22.93", "10.00"}


def test_unknown_scan_picture_is_404(tmp_path):
    c = client("50.00", "22.93", "2.18", store=ScanStore(tmp_path))
    assert c.get("/scans/nope/image").status_code == 404
    assert c.get("/scans/..%2Fscans.db/image").status_code == 404


def test_access_code_protects_every_route(tmp_path):
    c = client("50.00", "22.93", "2.18", store=ScanStore(tmp_path), access_code="correct horse")
    assert c.get("/").status_code == 401
    assert c.get("/scans").status_code == 401
    assert post(c).status_code == 401
    assert c.get("/", auth=("me", "wrong")).status_code == 401
    assert "Basic" in c.get("/").headers["www-authenticate"]
    assert ScanStore(tmp_path).recent() == []  # the refused scan was not stored

    c.auth = ("anyone", "correct horse")
    assert c.get("/").status_code == 200
    assert post(c).json()["liters"] == "22.93"


def test_repeated_wrong_codes_lock_everyone_out_for_a_while(monkeypatch):
    import app.main as main

    clock = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    c = client("50.00", "22.93", "2.18", access_code="correct horse")
    right, wrong = ("me", "correct horse"), ("me", "guess")

    for _ in range(main.MAX_FAILED_LOGINS - 1):
        assert c.get("/", auth=wrong).status_code == 401
    assert c.get("/", auth=right).status_code == 200  # not locked yet

    assert c.get("/", auth=wrong).status_code == 401  # the last allowed failure
    locked = c.get("/", auth=right)
    assert locked.status_code == 429
    assert locked.headers["retry-after"] == str(main.LOCKOUT_SECONDS)
    assert c.get("/", auth=wrong).status_code == 429

    clock[0] += main.LOCKOUT_SECONDS + 1
    assert c.get("/", auth=right).status_code == 200


def test_visits_without_any_code_do_not_count_towards_the_lockout():
    import app.main as main

    c = client("50.00", "22.93", "2.18", access_code="correct horse")
    for _ in range(main.MAX_FAILED_LOGINS + 5):
        assert c.get("/").status_code == 401
    assert c.get("/", auth=("me", "correct horse")).status_code == 200


def test_production_app_refuses_to_start_without_an_access_code(monkeypatch, tmp_path):
    monkeypatch.setenv("PUMP_OCR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PUMP_OCR_ACCESS_CODE", raising=False)
    with pytest.raises(RuntimeError, match="PUMP_OCR_ACCESS_CODE"):
        production_app()
    monkeypatch.setenv("PUMP_OCR_ACCESS_CODE", "short")
    with pytest.raises(RuntimeError):
        production_app()


def test_large_photos_are_shrunk():
    image = decode_image(jpeg(width=4032, height=3024))
    assert max(image.shape[:2]) == MAX_SIDE
    assert decode_image(jpeg()).shape == (300, 400, 3)
    assert decode_image(b"") is None


def test_two_row_display_is_returned_as_review_and_stored(tmp_path):
    store = ScanStore(tmp_path)
    c = client("57.62", "33.899", store=store)
    body = post(c).json()
    assert (body["status"], body["liters"], body["flags"]) == ("review", "33.899", ["no_price_row"])
    (saved,) = store.recent()
    assert (saved["status"], saved["liters"]) == ("review", "33.899")
    assert "33.899 L (not verified)" in c.get("/scans").text


def test_each_person_signs_in_with_their_own_code_and_scans_are_labelled(tmp_path):
    store = ScanStore(tmp_path)
    users = {"sara": "sara-code-1234567", "ahmed": "ahmed-code-123456"}
    c = client("50.00", "22.93", "2.18", store=store, users=users)

    assert post(c).status_code == 401
    assert c.get("/", auth=("sara", "not-a-code")).status_code == 401

    # The code identifies the person; the typed user name does not matter.
    c.auth = ("whatever", "ahmed-code-123456")
    post(c)
    c.auth = ("sara", "sara-code-1234567")
    post(c)

    assert sorted(s["user"] for s in store.recent()) == ["ahmed", "sara"]
    page = c.get("/scans").text
    assert "by ahmed" in page and "by sara" in page


def test_shared_code_and_personal_codes_work_together(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store, access_code="shared-code", users={"sara": "sara-code"})
    c.auth = ("x", "shared-code")
    post(c)
    assert store.recent()[0]["user"] == "shared"


def test_scans_without_sign_in_have_no_user(tmp_path):
    store = ScanStore(tmp_path)
    post(client("50.00", "22.93", "2.18", store=store))
    assert store.recent()[0]["user"] is None


def test_database_from_before_users_is_upgraded(tmp_path):
    import sqlite3

    db = sqlite3.connect(tmp_path / "scans.db")
    db.execute(
        "CREATE TABLE scans (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,"
        " liters TEXT, confidence REAL, flags TEXT NOT NULL, scan_seconds REAL NOT NULL,"
        " image_file TEXT NOT NULL)"
    )
    db.execute("INSERT INTO scans VALUES ('old', '2026-10-06T08:36:19+00:00', 'retake', NULL, NULL, '[]', 1.0, 'old.jpg')")
    db.commit()
    db.close()

    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store, users={"sara": "sara-code"})
    c.auth = ("sara", "sara-code")
    post(c)
    by_id = {s["id"]: s["user"] for s in store.recent()}
    assert by_id.pop("old") is None
    assert list(by_id.values()) == ["sara"]


def test_user_file_round_trip_and_generated_codes(tmp_path):
    path = tmp_path / "users.yaml"
    assert load_users(path) == {}
    codes = {generate_code() for _ in range(50)}
    assert len(codes) == 50 and all(len(code) == 23 for code in codes)
    users = {"sara": generate_code(), "ahmed": generate_code()}
    save_users(path, users)
    assert load_users(path) == users


def test_two_users_with_the_same_code_are_refused(tmp_path):
    path = tmp_path / "users.yaml"
    save_users(path, {"sara": "same", "ahmed": "same"})
    with pytest.raises(ValueError, match="same code"):
        load_users(path)


def test_a_pin_only_works_with_its_owners_name(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store, users={"sara": "48213", "ahmed": "90357"})
    assert c.get("/", auth=("anyone", "48213")).status_code == 401
    assert c.get("/", auth=("ahmed", "48213")).status_code == 401  # sara's PIN, ahmed's name
    c.auth = (" Sara ", "48213")  # case and stray spaces are forgiven
    assert c.get("/").status_code == 200
    post(c)
    assert store.recent()[0]["user"] == "sara"


def test_pins_get_the_stricter_lockout(monkeypatch):
    import app.main as main

    clock = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    c = client("50.00", "22.93", "2.18", users={"sara": "48213"})
    for _ in range(main.SHORT_CODE_MAX_FAILED_LOGINS):
        assert c.get("/", auth=("sara", "00000")).status_code == 401
    assert c.get("/", auth=("sara", "48213")).status_code == 429

    clock[0] += main.LOCKOUT_SECONDS + 1  # the long-code window is not enough
    assert c.get("/", auth=("sara", "48213")).status_code == 429
    clock[0] += main.SHORT_CODE_LOCKOUT_SECONDS
    assert c.get("/", auth=("sara", "48213")).status_code == 200


def test_long_personal_codes_still_work_with_any_user_name():
    c = client("50.00", "22.93", "2.18", users={"sara": "k7m2p-x9qrt-4hwne-j3bvf"})
    assert c.get("/", auth=("whatever", "k7m2p-x9qrt-4hwne-j3bvf")).status_code == 200


def test_generated_pins_are_five_digits_unique_and_not_obvious():
    taken = set()
    for _ in range(300):
        pin = generate_pin(taken)
        assert len(pin) == 5 and pin.isdigit() and pin not in taken
        assert len(set(pin)) > 1 and pin not in ("12345", "54321", "01234")
        taken.add(pin)


def scan_once(c):
    return post(c).json()["id"]


def test_confirming_a_result_records_its_liters_as_true(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store)
    scan_id = scan_once(c)

    reply = c.post(f"/scans/{scan_id}/feedback", json={"correct": True})
    assert reply.json() == {"id": scan_id, "feedback": "correct", "true_liters": "22.93"}
    assert store.get(scan_id)["true_liters"] == "22.93"
    assert "confirmed correct" in c.get("/scans").text


def test_marking_a_result_wrong_records_what_the_pump_showed(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store)
    scan_id = scan_once(c)

    reply = c.post(f"/scans/{scan_id}/feedback", json={"correct": False, "true_liters": "22,98"})
    assert reply.json() == {"id": scan_id, "feedback": "wrong", "true_liters": "22.98"}
    assert "marked wrong, pump showed 22.98 L" in c.get("/scans").text

    # The true value is optional: "wrong, but I don't know what it was".
    other = scan_once(c)
    assert c.post(f"/scans/{other}/feedback", json={"correct": False}).json()["true_liters"] is None


def test_after_a_retake_the_true_liters_can_be_typed(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "888.18", store=store)  # a retake
    scan_id = scan_once(c)

    assert c.post(f"/scans/{scan_id}/feedback", json={"correct": False}).status_code == 422
    reply = c.post(f"/scans/{scan_id}/feedback", json={"correct": False, "true_liters": "22.93"})
    assert reply.json() == {"id": scan_id, "feedback": None, "true_liters": "22.93"}
    assert "pump showed 22.93 L" in c.get("/scans").text


@pytest.mark.parametrize("typed", ["abc", "22.93 L", "1234.5", "-5", "22..9", "<b>"])
def test_typed_liters_must_be_a_number(tmp_path, typed):
    c = client("50.00", "22.93", "2.18", store=ScanStore(tmp_path))
    scan_id = scan_once(c)
    reply = c.post(f"/scans/{scan_id}/feedback", json={"correct": False, "true_liters": typed})
    assert reply.status_code == 422


def test_feedback_on_an_unknown_scan_is_404(tmp_path):
    c = client("50.00", "22.93", "2.18", store=ScanStore(tmp_path))
    assert c.post("/scans/nope/feedback", json={"correct": True}).status_code == 404
    assert client("50.00", "22.93", "2.18").post("/scans/x/feedback", json={"correct": True}).status_code == 404


def test_only_the_person_who_scanned_can_give_feedback(tmp_path):
    store = ScanStore(tmp_path)
    users = {"sara": "sara-code-1234567", "ahmed": "ahmed-code-123456"}
    c = client("50.00", "22.93", "2.18", store=store, users=users)
    c.auth = ("x", "sara-code-1234567")
    scan_id = scan_once(c)

    c.auth = ("x", "ahmed-code-123456")
    assert c.post(f"/scans/{scan_id}/feedback", json={"correct": True}).status_code == 404
    c.auth = ("x", "sara-code-1234567")
    assert c.post(f"/scans/{scan_id}/feedback", json={"correct": True}).status_code == 200


def test_page_has_the_feedback_buttons():
    page = client("50.00", "22.93", "2.18").get("/").text
    assert 'id="feedback-yes"' in page and 'id="true-liters"' in page



def test_open_scanning_needs_no_sign_in_but_the_history_does(tmp_path):
    store = ScanStore(tmp_path)
    c = client("50.00", "22.93", "2.18", store=store, users={"sara": "48213"}, open_scanning=True)
    assert c.get("/").status_code == 200
    scan = post(c)
    assert scan.status_code == 200 and scan.json()["liters"] == "22.93"
    assert store.recent()[0]["user"] is None
    assert c.post(f"/scans/{scan.json()['id']}/feedback", json={"correct": True}).status_code == 200
    # what other people scanned stays behind the code
    assert c.get("/scans").status_code == 401
    assert c.get(f"/scans/{scan.json()['id']}/image").status_code == 401
    assert c.get("/scans", auth=("sara", "48213")).status_code == 200
