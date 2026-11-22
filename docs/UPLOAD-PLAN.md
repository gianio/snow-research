# Upload plan – what you upload, when, and how

Goal for the next weeks: steer the model by three numbers (dashboard section «Trend»):

| Number | Target | Reference | Who provides it |
|---|---|---|---|
| New snow 24 h: model ÷ SLF map, error on 10 km blocks | 0.9–1.1, < 2 cm | SLF new-snow map HN1D | automatic (map is public back to 2022) |
| Snow height Variant A: bias at holdout stations | ±2 cm | IMIS stations Variant A never uses | automatic (Variant A manifest, archived every cycle) |
| Snow surface: hit rate / share with a model class | ≥ 70 % / ≥ 90 % | SLF Pro observer symbols | **you: screenshots** |

## Automatic – nothing to do

* **Model archive** (`archive-forecasts.yml`, every hour): every Snowmapper run and every Variant A cycle is saved as a release asset (`fc-archive-YYYY-MM`). Release assets are kept until deleted.
  Needs the repository secret `MODEL_SITE` (= address of the Snowmapper site, the folder with `data/latest.json`).
  If a run fails, GitHub e-mails you.
* SLF new-snow maps and the Variant A holdout numbers are read when the dashboard is updated.

## Every day, in the evening (after about 18:00)

By then the observers' morning new snow and the day's surface reports are in.

| # | Screenshot | SLF Pro view | File name |
|---|---|---|---|
| 1 | Observed new snow | Beobachtungen → Neuschnee, today | `obs_hn_YYYY-MM-DD.png` |
| 2 | Snow surface | Beobachtungen → Schneeoberfläche, today | `surface_YYYY-MM-DD.png` |

Missed a day? Catch up within a week: SLF Pro lets you pick past dates, and the model runs for those days are in the archive.

Optional, after a snowfall of more than 20 cm: WhiteRisk new-snow 72 h area map (`wr_hn72_map_YYYY-MM-DD.png`).
Not needed: WhiteRisk station screenshots (the IMIS stations give those values exactly, and reading them from screenshots is unreliable).

### How the screenshot must look

* Desktop browser, **always the same window size and zoom** (then the settings in the manifest keep working).
* All of Switzerland visible including the whole border, legend visible, no pop-up or tooltip open.
* PNG, at least 2000 px wide (a full-window screenshot on a normal screen is fine). No phone screenshots.

### Where

`screencheck/inbox/YYYY-MM-DD/` (date of the upload), or a Google Doc / .docx / chat like before.
Then I run `screencheck.py` and `dashboard_data.py` and update the dashboard.

## Once a week (e.g. Sunday evening)

Send me the week's screenshots if you haven't yet. I check them, update the dashboard and tell you which numbers moved.

## Every time you change the model

Add one line to `docs/MODEL-CHANGES.md`: date and time (UTC), what changed, which number you expect to move.
Change **one thing at a time** and judge it only after at least three days with snow; the archive lets the dashboard compare the old and the new run on the same days.
