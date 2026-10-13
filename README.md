# Pump display OCR (prototype)

Backend prototype that reads the liters value from photos of Saudi gas pump
displays and returns it only once it is validated. The total (SAR) and the
price per liter are read too, but only to check the liters value: liters x
price per liter must equal the total. OCR runs locally with PaddleOCR; images
are never sent to an external service.

**Status:** step 5 of 8. Screen cropping, OCR, parsing, validation, the
evaluation script and a development phone-scan server exist. No fraud checks
or production API yet.

**Focus: liters only.** The app returns the liters value and nothing else.
The original API contract returned amount, volume and price; `POST /readings`
will need its response updated to match when it is built.

## Setup (Windows)

Requires [uv](https://docs.astral.sh/uv/). The virtual environment lives
outside the project folder because the project is inside OneDrive, and
OneDrive syncing a venv causes file locks.

```powershell
uv venv C:\venvs\pump-ocr --python 3.11
uv pip install --python C:\venvs\pump-ocr\Scripts\python.exe -r requirements.txt
```

Commands below call the venv's Python by full path, so no activation is
needed. To activate in PowerShell anyway, the `.ps1` extension is required
(`C:\venvs\pump-ocr\Scripts\activate.ps1`); without it nothing happens.

The first run downloads two OCR models (detection and recognition) to
`%USERPROFILE%\.paddlex\official_models`. If the download times out, switch
the model source and run again:

```powershell
$env:PADDLE_PDX_MODEL_SOURCE = 'huggingface'
```

## Run OCR on one image

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\ocr_one.py data\some_photo.jpg
```

Prints `accepted` with the liters value, or `retake` with the reasons and
what each attempt read. Rows are identified by position, not by the letters
printed on the display. Several photos can be given at once. Takes about 10
seconds to load the models, then about 3 seconds per photo.

- `--raw` also prints every raw OCR detection (confidence, text, box).
- `--save-crops` saves the region the OCR was given as `<photo>_screen.png`
  next to each photo, to check by eye that the display was targeted.

## How a reading is accepted

`app/pipeline.py` runs these steps and returns `accepted` or `retake`:

1. Crop the photo to the display (or use the whole photo if no screen
   outline is found), then shrink the crop to at most `ocr_max_side` pixels.
2. Run OCR on up to four versions of the crop (original, faded, binary,
   and one with thickened digit strokes), stopping at the first one that
   yields an accepted result if `stop_at_first_valid` is on: the original, one with faint
   pixels faded to white, and a pure black-and-white one. LCDs show their
   unlit segments faintly, and the original often reads them as digits
   (`988.99` for `1.99`); the cleaned versions avoid that but can erase thin
   strokes.
3. Parse each version into amount, volume, price.
4. Try every combination of the amounts, volumes and prices that were read
   against the validation rules.
5. Accept only if exactly one combination passes. None passing, or two
   different ones passing, is a retake. A retake never returns values.
6. Each of the three values must have an OCR confidence of at least
   `min_confidence` (in `config/settings.yaml`), otherwise retake.
7. If `confirm_liters_row` is on, cut out just the liters row and read it
   again on its own. It must show the same liters value, otherwise retake.
8. Return the liters value only.

OCR confidence is not proof of correctness: a ghost-segment misread
(`8882.33`) scored 1.000. The validation rules are what stop wrong values;
the confidence floor is an extra filter on top.

Validation rules (`app/validation/rules.py`), all limits in `config/`:

- amount has exactly 2 decimals
- price equals a grade in `config/fuel_prices.yaml` **(switched off for now:
  `check_fuel_price: false` in `config/settings.yaml`; the code and the price
  list are kept)**
- `abs(volume * price - amount) <= 0.02` SAR
- volume between 1 and 150 L
- volume has 2 or 3 decimals (`volume_decimals`, depends on the pump)

To add or change a fuel price, edit `config/fuel_prices.yaml`.

## How screen detection works

`app/ocr/preprocess.py` looks for the largest roughly rectangular area that
is clearly whiter than its surroundings (backlit LCD: bright and nearly
colourless, so a bright blue or red pump body is not mistaken for it) or
clearly darker (LED display) and crops to it, so stickers and signs around the pump do not
reach the OCR. If no such area is found (the display fills the photo, or it
has little contrast against the pump body) the whole photo is read instead.

Use the venv's Python as shown. A plain `python` or the VS Code Run button may
use a different interpreter and fail with `No module named 'paddleocr'`.

## Scan from a phone

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\phone_scan.py
```

The script prints an address such as `https://192.168.8.232:8000`. Open it in
the browser of a phone on the same Wi-Fi. The camera fills the whole screen,
with a box in the middle (the region of interest) and the **Scan** button at
the bottom; nothing needs scrolling. Fit the pump display inside the box and
tap **Scan**. The picture is taken at the moment of the tap and the view
freezes, so the phone can be lowered straight away. Only the box, plus a
small margin, is sent to the server. The
result slides up above the button and prints in the terminal in the same
format as `ocr_one.py`. Ctrl+C stops the server.

- **Zoom.** Pinch the picture or use the slider above the button.
- **Arabic / English.** The round button at the top (ع / EN) switches every
  message on the page, and Arabic is laid out right to left. The choice is
  remembered on that phone; a phone set to Arabic starts in Arabic. The
  words are in `TEXT` in `app/static/scan.html`. The history page is in
  English only.
- **Arabic words on the pump** are listed in `config/labels.yaml` but are
  not active: the reader's alphabet has no Arabic letters. Pumps that print
  English next to the Arabic are unaffected.

- **Scan time.** The button counts up while a scan runs and the result says
  how long it took from tap to result. The terminal prints the reading time
  as `Scan time: 1.23 s`.
- **Certificate warning.** Phones only allow the live camera over HTTPS, so
  the script creates a self-signed certificate in `data\dev_cert\`. The phone
  warns that the connection is not private; in Safari tap **Show Details**,
  then **visit this website**, then allow camera access.
- `--http` serves plain HTTP instead: no warning, but no live view and no
  box. The page then falls back to the phone's own camera app.
- To change the box, edit `--roi` (its size) and `#roi` in
  `app/static/scan.html`.
- The first time, Windows Firewall asks whether to allow Python; allow it on
  private networks or the phone cannot connect.
- **Every scan is stored**: the picture in `data\scans\` and the result
  (time, status, liters, confidence, retake reasons) in `data\scans.db`.
  Retakes are stored too. Open `/scans` on the server for the history page.
- **Access code.** Add `--access-code "a long secret"` to require it; the
  browser asks once (any user name, the code as password). Without it,
  anyone on the same network can use the server while it runs.
- None of the fraud checks exist yet, and `POST /readings` is not built.

## Scan from anywhere with Tailscale (laptop stays at home)

Tailscale gives the phone a private, encrypted link to this PC, so scanning
works over mobile data while the PC stays at home. Free; nothing is exposed
to the public internet.

One-time setup:

1. Install Tailscale on the PC and the phone, signed in to the same account.
2. Enable "Serve" for the PC: run `tailscale serve --bg http://127.0.0.1:8000`
   once. The first time it prints a link; open it and approve, then run the
   command again.
3. Set the PC not to sleep while plugged in.

Each time:

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\phone_scan.py --tailscale --access-code "a long secret"
```

Open the printed `https://<pc-name>.<tailnet>.ts.net` address on the phone
(Tailscale app connected). Tailscale supplies a real certificate, so there is
no warning. In this mode the server only listens on the PC itself; nothing
on the local Wi-Fi can reach it.

`tailscale serve reset` turns the forwarding off again.

### Letting someone else scan: public mode (Tailscale Funnel)

Funnel makes the same address reachable from any phone, with no Tailscale
app. The other person only needs the link and the access code.

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\phone_scan.py --public --access-code "at least 16 characters"
& "C:\Program Files\Tailscale\tailscale.exe" funnel --bg http://127.0.0.1:8000
```

The first `funnel` run prints a link to approve Funnel for the account.
`tailscale funnel reset` makes the address private again.

- **The server is then on the public internet.** Anyone who learns the
  address can reach the login prompt; the access code is the only
  protection, which is why `--public` insists on 16 characters or more.
- **Lockout:** after 10 wrong codes within 5 minutes, every request is
  refused (including ones with the right code) until 5 minutes have passed.
  Someone hammering the login can therefore keep legitimate users out, but
  cannot keep guessing.
- Give each person their own code (see "Users and personal access codes")
  so the history shows who scanned.
- Turn Funnel off when it is not needed.

## Hosting on a Linux server

**Not built or tested yet**: `Dockerfile` and `docker-compose.yml` were
written on a machine without Docker.

The compose file runs two containers: the OCR server, and Caddy, which
provides HTTPS with an automatically issued certificate (phones only allow
the camera page over HTTPS).

1. Get a Linux server (4 CPU cores, 8 GB RAM is a reasonable start) with
   Docker installed, and a domain name pointing at it.
2. Copy the project to the server.
3. Create a `.env` file next to `docker-compose.yml`:
   ```
   PUMP_OCR_ACCESS_CODE=a-long-random-secret
   PUMP_OCR_DOMAIN=scan.example.com
   ```
4. `docker compose up -d --build`
5. Open `https://scan.example.com` on the phone and sign in with any user
   name and the access code as password.

- The server refuses to start without an access code of at least 12
  characters, so it cannot be on the internet unprotected.
- Scans are kept in the `scans` Docker volume and survive restarts.
- The OCR models are downloaded into the image at build time; the running
  server needs no outbound internet access.
- There is no limit on wrong access-code attempts and one shared code for
  everyone. Use a long random code. Per-user accounts are not built.
- PDPL: stored pictures and their timestamps are personal data once tied to
  a user. Choose a server location, a retention period and a deletion
  process before real users are involved.

## Users and personal access codes

Each person can have their own access code, so the history shows who made
each scan and one person can be removed without changing everyone's code.

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\manage_users.py add sara ahmed
C:\venvs\pump-ocr\Scripts\python.exe scripts\manage_users.py list --show-codes
C:\venvs\pump-ocr\Scripts\python.exe scripts\manage_users.py rename user1 sara
C:\venvs\pump-ocr\Scripts\python.exe scripts\manage_users.py remove ahmed
C:\venvs\pump-ocr\Scripts\python.exe scripts\manage_users.py new-code sara
```

- Codes are generated (20 random characters in four groups) and stored in
  `data\users.yaml`. That file holds them in plain text: keep it private.
- A person signs in with any user name and their own code as the password.
  The code alone identifies them.
- `phone_scan.py` loads `data\users.yaml` automatically; `--access-code` is
  then optional. Restart the server after any change to the users.
- Scans made before a person had a code, or with the shared `--access-code`,
  show no user or `shared`.

### Scanning without signing in

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\phone_scan.py --public --allow-short-codes --no-login
```

With `--no-login` anyone who has the address can open the scan page and
scan; nobody is asked for a name or code, and scans are stored without a
user name. The history page (`/scans`) and the stored pictures still ask
for a user name and code, because they show what other people scanned.

Open means open: anyone on the internet who finds the address can upload
pictures, which the laptop stores and reads. For a short trial only. Stop
the server when the trial is over.

### 5-digit PINs

`manage_users.py add NAME --pin` or `new-code NAME --pin` gives a 5-digit PIN
instead of a long code. PINs are easier to type and much easier to guess, so
they work differently:

- The person must type **their own name** as the user name (capitals do not
  matter) and the PIN as the password. A PIN with someone else's name, or
  with no matching name, is refused.
- The lockout is stricter while any user has a PIN: 5 wrong tries, then
  everyone is locked out for 15 minutes.
- `--public` refuses to start with PINs unless `--allow-short-codes` is
  added. With that lockout an attacker who knows a user's name gets at most
  480 guesses a day out of 100,000 PINs, roughly a 1 in 200 chance per day
  per known name. Someone who gets in can scan and see the scan history,
  nothing else. Fine for a short test among colleagues; not for real use.
- Anyone hammering the login keeps everyone else locked out too.

## Run a whole folder of photos

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\batch.py "data\Pump Displays_Large"
C:\venvs\pump-ocr\Scripts\python.exe scripts\batch.py "data\Pump Displays_Large" --csv results.csv
```

Prints one line per photo (status, liters, time, what it was checked
against or why not) and a summary. No labels are needed: it shows what the
app returns, not whether that is right. `--limit N` and `--only NAME ...`
run a subset.

## "Was this right?": labelled data from everyday scans

After each scan the phone asks whether the result was right:

- **Liters returned:** "Is 22.93 what the pump shows?" with **Yes** and
  **No**. After **No**, the person can type what the pump showed (optional).
- **Retake:** an optional box to type the liters the pump showed.

The answer is stored with the scan and shown on the history page. Only the
person who made a scan can answer for it.

Turn the answers into a test set and measure the app on real scans:

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\export_feedback.py
C:\venvs\pump-ocr\Scripts\python.exe scripts\evaluate.py data\scans\labels.csv
```

- These labels have only the liters, which is all `evaluate.py` needs.
- They are only as good as what people typed. A "wrong liters returned" in
  the report can be a typing mistake: open the picture before believing it.
- The same labelled scans are what the reader should be retrained on later
  (`scripts/train/README.md`). Keep the ones used for training apart from
  the ones used for measuring.

## Check what is stored

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\check_storage.py
```

Lists every stored scan (time, status, liters, picture file) and checks that
each record has its picture and each picture has its record. Add `--open` to
open the newest picture. A picture and its record are saved as one unit: if
the record cannot be written, the picture is removed again.

## Evaluate against labelled photos

Put the photos and a `labels.csv` in `data/`. The CSV has one row per photo
with the values a person read off the display:

```csv
filename,amount,volume,price
pump_0002.jpg,57.75,26.49,2.18
```

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\evaluate.py data\labels.csv
```

Headline: how often the liters value is returned correctly, and how many
WRONG liters values were returned (target: zero). Also reports accepted /
retake / review counts, retake reasons, liters misreads in individual reads
(digit swaps, extra or missing characters) and how accurate each image
version is per row. Exits with code 1 if any wrong liters value was returned.
Run it after every change to the pipeline or to `config/`.

## Real display photos from the internet

`data\internet_displays\` holds real photos of pump displays that show the
liters pumped, found with `scripts\find_display_photos.py` and each checked
by eye. They include Gulf displays with Arabic labels.

- `commercial_ok\`: licences that allow any use with credit (CC BY, BY-SA).
- `testing_only\`: non-commercial or no-changes licences. Fine for testing
  here; **do not ship them or train a commercial model on them.**
- `sources.csv` in each folder gives the page, licence and author.
- `labels.csv` has the liters read by eye. Pictures showing two displays at
  once have no label.

```powershell
C:\venvs\pump-ocr\Scripts\python.exe scripts\evaluate.py data\internet_displays\labels.csv
```

To look for more: `search` collects small previews, a person picks the ones
that really show a display, `download` fetches those in full.

```powershell
C:\venvs\pump-ocr\Scripts\python.exe -X utf8 scripts\find_display_photos.py search --work C:\ocr-train\internet_candidates
C:\venvs\pump-ocr\Scripts\python.exe -X utf8 scripts\find_display_photos.py download --work C:\ocr-train\internet_candidates --keep 17 73 --out data\internet_displays\testing_only
```

Only the picture search goes to the internet. Nothing from `data\` is sent.

## Run the tests

```powershell
C:\venvs\pump-ocr\Scripts\python.exe -m pytest
```

## How parsing works

`app/ocr/parse.py` looks for exactly three decimal numbers stacked in one
column and reads them top to bottom as amount, volume, unit price. It never
corrects text: a number with a missing decimal point or a letter in it is not
used, and anything ambiguous is a parse failure (which will become "retake").

## The trained number reader

PaddleOCR's stock recognition model was trained on printed text. On pump
displays it drops decimal points and misreads 7-segment digits. The app uses
a version fine-tuned on pump numbers, `models/pump_rec_v1` (set in
`config/settings.yaml`, `ocr` section), and runs it on the graphics card.
How it was trained, and how to train again: `scripts/train/README.md`.

Measured on 2026-10-06 on photos never used for training:

| Photo set | Stock model, CPU | Trained model, GPU |
|---|---|---|
| Large (140 rendered pumps): liters correct | 78 (56%) | 123 (88%) |
| Large: wrong liters returned | 0 | 0 |
| Large: accepted / review / retake | 24 / 54 / 62 | 115 / 8 / 17 |
| Large: time for all 140 | about 20 min | about 1 min |
| Small (12 close-ups): accepted | 12 | 12 |
| Phone scans stored that day (18): liters returned | 7 | 10 |

Limits of that measurement:

- The Large and Small photos come from the same image generator as most of
  the training pictures. The score shows the model learned that pump style.
  It does not prove accuracy on real pumps; only labelled real photos can.
- The model reads label words slightly worse than before (`TOTAL` came out
  as `TOTAL1` once). The parser tolerates one stray digit in a word.
- `models/` is not in git (73 MB). Keep a copy; retraining takes about 2.5
  hours on an RTX 3070.
- Without an NVIDIA card set `ocr.device: cpu`. It works, several times
  slower. The Docker image runs on the CPU.

## Telling the numbers apart: position and labels

`app/ocr/layout.py` decides which number is the total, the liters and the
price per liter from three kinds of evidence:

- **Size.** The total and the liters are the biggest numerals on a pump.
  The tallest number, plus the numbers stacked with it at a similar size,
  form the main column. Small digits elsewhere (keypad keys, the pump
  number, stickers) are ignored. This is what lets a photo of the whole
  pump work, not only a close-up of the display.
- **Position.** Numbers stacked in one column are, top to bottom: total,
  liters, price per liter.
- **Labels.** Words next to a number say what it is: `SAR`, `L`, `LITER`,
  `SAR/L`, and so on. The word lists are parameters in `config/labels.yaml`;
  add the words used on the pumps you scan.

A label is used only when it clearly belongs to exactly one number (it is on
that number's row and close to it). Then it wins over position. That is what
makes other layouts work:

- rows in another order (for example liters above the total)
- the price per liter in a separate panel, away from the other two numbers
- a label stuck to its number by the OCR (`SAR179.85`, `4.59L`)

A label that could belong to two rows is ignored, never guessed. A display
whose words start with `GAL` is refused, because its number is gallons.

This is a correctness matter: liters x price = total holds equally with
liters and price swapped, so the arithmetic check cannot tell them apart.
With the Saudi price rule switched off, only position and labels can. A pump
that orders its rows differently and has no readable labels would return the
price as the liters, marked accepted.

**Whole-photo fallback.** If the cropped display gives nothing (the crop can
be the wrong area, or the pump can have several panels), the whole photo is
read as well (`whole_photo_fallback` in `config/settings.yaml`). This adds
several seconds to scans that would otherwise be a retake.

**Reading at several sizes.** Digits that are very tall in the picture (a
close-up) are not found: each row is cut into a top and a bottom half, or
nothing is found at all. The same picture shrunk reads correctly. So a
picture that gives no reading is read again at the sizes in
`extra_ocr_sides` (the cropped display) and `extra_whole_photo_sides` (the
whole photo). The first size that gives liters wins. A two-row result is
refused if another size saw a different number under the same total.

## Restored decimal points ("review")

The OCR often misses the small decimal point of a 7-segment number and reads
`10841` for `108.41`; zooming in does not help. With `infer_decimal_points`
on, a number read without a point is given its usual decimals: 2 for the
total and the price per liter, `inferred_liters_decimals` (2) for liters.

Such a result is never `accepted`. It is returned as `review` with the flag
`decimal_point_inferred`, and only if the check still holds: liters x price
= total when a price was read, the plausible-price range otherwise. A total
that lost its last digit (`1084` for `10841`) fails that check.

Known gap: if the total and the liters both lose their last digit in the
same photo, the arithmetic still holds and a value ten times too small would
be returned as review. Set `infer_decimal_points: false` to rule this out,
at the cost of many more retakes on displays whose points are not read.

## Displays without a price per liter ("review")

Some pumps show only two rows: the total and the liters. With no price row
the arithmetic check is impossible, so such a scan is never `accepted`. If
`allow_two_row_displays` is on (it is), it returns liters with status
`review` and the flag `no_price_row`, meaning: read, but not verified.

A review needs all of these:

- no image version saw a third number, or any other digits, in the column
- every version that read two numbers read the same two
- liters was read twice: by two image versions, or by one version plus the
  re-read of the liters row (two agreeing versions need no re-read)
- liters passes the range and decimals rules, the total has 2 decimals
- total / liters is between `two_row_min_price` and `two_row_max_price`
  (catches the two numbers being liters and price instead of total and
  liters)

A review is a weaker result than accepted: a misread digit in liters that
both reads repeat would not be caught. Treat review values as needing a
human look at the stored picture.

## Switched off for now

- **Saudi fuel price rule.** `check_fuel_price: false` in
  `config/settings.yaml`. Pumps showing any price per liter are accepted as
  long as liters x price per liter equals the total. With the rule off, the
  price is no longer checked against a known list, so the arithmetic check
  is slightly weaker than with it on. Set it to `true` to restore.
- **Total must have 2 decimals.** `amount_decimals: [2, 3]` also allows 3,
  for currencies such as the Kuwaiti dinar. Set to `[2]` to restore.
- **Saudi price range on two-row displays.** `two_row_min_price` and
  `two_row_max_price` are wide (0.05 to 150) so any currency passes. With
  them wide, a two-row "review" result rests on the liters being read twice
  with the same value, and little else. Set to 0.5 and 5 to restore.

All three must go back before this is used for real Saudi readings.

## Stubs

None yet. App Attest and signature verification will be added as stubs behind
config flags that are off in production mode.

## Known limitations

- **Reading is unreliable on display styles it was not built around.** In
  the first office trial (2026-10-06) five scans of European and US pump
  pictures shown on a monitor all came back as retake. After fixes, one of
  the five reads. The rest fail at the OCR itself: PaddleOCR is a general
  text model, and faint digits, slanted or outlined 7-segment digits, light
  digits on a mid-tone background, and the pixel grid of a photographed
  monitor defeat it. No wrong value was returned in any of these cases. The
  real fix is an OCR model trained on 7-segment digits (not done).
- **Units are not read.** A US pump showing gallons is returned as if the
  number were liters.
- **Price shown in cents** (German pumps: `145,9` cents per liter) is not
  converted, so such a three-number display fails the arithmetic check.

- **Windows Smart App Control blocks this project when it is on.** It
  refuses unsigned files such as the venv's `python.exe` and OpenCV's
  library ("An Application Control policy has blocked this file"). The
  project then cannot run on that PC at all; it has to run on Linux (the
  hosted server) or Smart App Control has to be turned off in Windows
  Security.

- **paddlepaddle is pinned to 3.2.2, not the latest.** 3.3.1 crashes in its
  oneDNN backend on Windows CPU, and without oneDNN reads are about 1.7x
  slower. Re-test oneDNN before upgrading.
- **Speed is CPU-bound.** A scan takes about 3 s with the liters row
  re-read on (about 1.5 s without it), and longer when the first read fails
  and the other versions are needed. Running
  the OCR on a GPU would cut this to a fraction of a second but is not set
  up.
- **Early stop skips the cross-check.** With `stop_at_first_valid: true` a
  photo whose first read passes validation is accepted without the other two
  reads, so the "two different valid results" retake cannot trigger for it.
  Set it to `false` in `config/settings.yaml` for the stricter behaviour.
- **Tested on 13 photos only**, with no labelled ground truth yet. 11 are
  accepted, 1 is below the confidence floor and 1 shows a price that is not
  in the config.
- **Screen detection is a best-effort crop, not a check.** If no screen
  outline is found the whole photo is read, so text around the pump can
  reach the OCR. A larger bright or dark rectangle elsewhere in the shot (a
  white sign, a window) can also be picked instead of the screen. Either way
  a value is only returned if it passes validation.
- **A misread last digit of the amount can pass validation.** The 0.02 SAR
  tolerance means an amount that is off by 0.01 can still satisfy the
  arithmetic check. Measure this once labelled photos exist.
- **Prices have no effective dates yet.** The config holds one price per
  grade; a photo taken before a price change would be rejected after it.
- **One display layout is assumed:** three numbers in a column, ordered
  amount, volume, price. Pumps with a different order or a side-by-side
  layout will fail to parse or, if the order differs, be mislabelled (the
  validation step will catch that as a retake).
- **Model download needs internet on first run.** Only models are downloaded;
  no image leaves the machine. The Docker image will bundle the models.
- **The project folder is inside OneDrive.** Anything placed in `data/` is
  synced to Microsoft's cloud. Move the project out of OneDrive before adding
  real photos if that matters for PDPL.
- The harmless startup messages `INFO: Could not find files for the given
  pattern(s)` and `No ccache found` come from paddle and can be ignored.
