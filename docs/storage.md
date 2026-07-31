# Storage model

All persistent state lives in one SQLite file (`checktheball.sqlite` by
default, overridable with the `CHECKTHEBALL_DB` env var). It is a server-side
resource: the FastAPI process owns the file, and every HTTP client shares it.
The file is gitignored; `init_db()` recreates the tables on first startup.

The file holds two kinds of data with opposite sharing needs.

## Cache tables are global on purpose

`games`, `plays`, and `player_stats_cache` cache public MLB data that is
identical for every user (today's schedule and box scores, situational splits,
the day's hitting lines). One user's lookup warms the cache for the next, so a
repeated lookup of the same cached data is served, within its freshness window,
without another API call. Scoping the cache per user would multiply upstream
calls by the user count for no benefit, so it stays global.

## Cache rows are swept, the log is not

A freshness window only decides whether a row is worth *reading*; nothing in
`cached_fetch` deletes it. Most keys are never requested twice (a finished
game's box score is fixed and asked for once), so a stale row is not overwritten
either, and the file grows without bound. Box scores dominate that growth: the
raw MLB payload carries each game three times over (per-player stats, id lists
into them, and pre-rendered display rows), so one row runs to tens of KB.

`sweep_cache(max_age_days=7)` deletes aged rows from the cache tables. It runs
at startup, to clear whatever expired while the process was down, and every 24
hours after that, since a server that stays up for months is exactly the one
whose cache has grown enough to matter. The periodic pass runs in a worker
thread so the blocking SQLite call does not stall request handling, and a
failure logs rather than killing the loop.

The eviction age and the freshness windows are separate clocks: the windows are
seconds ("can I use this row?"), the sweep is days ("is this old enough to throw
away?"). Deleting a row that is still wanted costs one refetch, which is what
makes the sweep safe to run unattended.

The sweep never touches `queries`. That table is the only data in the file that
is not a disposable copy.

## The query log is global today, scoped later

`queries` records every answered question with its grounding score. It is a
single global log now, which is what the eval needs to compute an aggregate
grounding rate. If the product grows accounts, per-user history does not mean a
separate database per user; it means a `user_id` (or `session_id`) column on
`queries` and a `WHERE user_id = ?` filter. One shared database, partitioned
logically.

`scripts/review_queries.py` reads that log and reports what it says: score
distribution and trend, per-tool usage with the mean score of answers that used
each tool, questions answered with no tool call (a missing-tool signal), the
failures worth reading, and those same failures formatted to paste into
`run_eval.py`. Read-only, and it honors `CHECKTHEBALL_DB`.

Rows carry no model or prompt version, so a change in the trend cannot be
attributed to a change you made. Treat it as descriptive until that is fixed.
The log also stores `tool_calls_made` rather than `tool_results`, which means a
historical answer cannot be re-scored offline when `grounding.py` changes.

## Scaling path

SQLite is a single-writer embedded database, which is the right fit for one
FastAPI process. Multiple server instances cannot safely share one SQLite file
over a network, so horizontal scaling means moving to a client/server database
such as Postgres. `core/db.py` isolates all storage behind `read_cache`,
`write_cache`, `cached_fetch`, `sweep_cache`, and `log_query` with parameterized
queries and no sport-specific imports, so that swap is a contained change rather
than a rewrite.

Deployment note: because the file is local to the process, any host with an
ephemeral filesystem (most container platforms) discards it on every deploy and
restart. The cache half does not care and refills in seconds, but the query log
would silently reset to zero, since `init_db()` recreates the tables and nothing
errors. Persisting it means a mounted volume with `CHECKTHEBALL_DB` pointed at
it, or the move to Postgres above.
