# Eval run history

One row per `scripts/run_eval.py` run. Claim extraction is a model call, so
scores vary between runs; read this as a range, not a single number.

| Timestamp (UTC) | Questions | Claims | Grounded | Claim-level | Mean/answer | Tool faithful | By category (mean/answer) |
|---|---|---|---|---|---|---|---|
| 2026-07-23 02:44 | 35/35 | 92 | 91 | 0.989 | 0.971 | n/a | ambiguous 1.00; comparison 1.00; compound 1.00; date_range 0.75; edge_case 1.00; fantasy 1.00; leaderboard 1.00; live 1.00; opinion 1.00; opponent 1.00; out_of_scope 1.00; pitching 1.00; projection 1.00; simple 1.00; split 1.00 |
| 2026-07-23 03:28 | 35/35 | 98 | 96 | 0.980 | 0.994 | 27/28 | ambiguous 1.00; comparison 1.00; compound 1.00; date_range 1.00; edge_case 1.00; fantasy 0.89; leaderboard 1.00; live 1.00; opinion 0.92; opponent 1.00; out_of_scope 1.00; pitching 1.00; projection 1.00; simple 1.00; split 1.00 |
| 2026-07-23 23:47 | 35/35 | 93 | 92 | 0.989 | 0.997 | 27/28 | ambiguous 1.00; comparison 1.00; compound 1.00; date_range 1.00; edge_case 1.00; fantasy 1.00; leaderboard 1.00; live 1.00; opinion 0.91; opponent 1.00; out_of_scope 1.00; pitching 1.00; projection 1.00; simple 1.00; split 1.00 |
| 2026-08-04 00:22 | 40/40 | 84 | 83 | 0.988 | 0.998 | 29/30 | ambiguous 1.00; comparison 1.00; compound 1.00; date_awareness 1.00; date_range 1.00; edge_case 1.00; fantasy 1.00; leaderboard 1.00; live 1.00; opinion 0.92; opponent 1.00; out_of_scope 1.00; pitching 1.00; projection 1.00; simple 1.00; split 1.00 |
