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

## The query log is global today, scoped later

`queries` records every answered question with its grounding score. It is a
single global log now, which is what the eval needs to compute an aggregate
grounding rate. If the product grows accounts, per-user history does not mean a
separate database per user; it means a `user_id` (or `session_id`) column on
`queries` and a `WHERE user_id = ?` filter. One shared database, partitioned
logically.

## Scaling path

SQLite is a single-writer embedded database, which is the right fit for one
FastAPI process. Multiple server instances cannot safely share one SQLite file
over a network, so horizontal scaling means moving to a client/server database
such as Postgres. `core/db.py` isolates all storage behind `read_cache`,
`write_cache`, `cached_fetch`, and `log_query` with parameterized queries and no
sport-specific imports, so that swap is a contained change rather than a
rewrite.
