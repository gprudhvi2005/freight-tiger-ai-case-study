# Eval report

- pipeline rows: 20 (17 flagged, 3 justified)
- labeled set: 27 rows (20 candidates, 7 negative controls)

## 1. sample replay (3 rows from sample_output_format_v2.csv)

- PASS Delhi-Jaipur 2024-11-11: all fields match sample
- PASS Ahmedabad-Mumbai 2025-01-20: all fields match sample
- PASS Mumbai-Pune 2025-09-15: all fields match sample
- all 3 checks passed

## 2. labeled verdicts (hand-labeled candidate route-weeks)

- PASS Delhi-Jaipur 2024-11-11: verdict Yes
- PASS Delhi-Jaipur 2024-11-18: verdict Yes
- PASS Ahmedabad-Mumbai 2025-01-20: verdict No (justified) via N002
- PASS Chennai-Bangalore 2025-02-24: verdict No (justified) via N001
- PASS Chennai-Bangalore 2025-03-03: verdict No (justified) via N001
- PASS Chennai-Bangalore 2025-03-10: verdict Yes
- PASS Chennai-Bangalore 2025-03-17: verdict Yes
- PASS Mumbai-Pune 2025-06-23: verdict Yes
- PASS Mumbai-Pune 2025-09-15: verdict Yes
- PASS Mumbai-Pune 2025-10-06: verdict Yes
- PASS Mumbai-Pune 2025-10-20: verdict Yes
- PASS Mumbai-Pune 2025-10-27: verdict Yes
- PASS Mumbai-Pune 2025-11-03: verdict Yes
- PASS Mumbai-Pune 2025-11-10: verdict Yes
- PASS Mumbai-Pune 2025-11-17: verdict Yes
- PASS Mumbai-Pune 2025-11-24: verdict Yes
- PASS Mumbai-Pune 2025-12-01: verdict Yes
- PASS Mumbai-Pune 2025-12-08: verdict Yes
- PASS Mumbai-Pune 2025-12-15: verdict Yes
- PASS Mumbai-Pune 2025-12-22: verdict Yes
- all 20 checks passed

## 3. negative controls (must-not-flag route-weeks)

- PASS Mumbai-Delhi 2025-05-12: correctly not flagged
- PASS Kolkata-Bhubaneswar 2025-06-09: correctly not flagged
- PASS Mumbai-Delhi 2024-07-29: correctly not flagged
- PASS Delhi-Jaipur 2024-05-20: correctly not flagged
- PASS Mumbai-Pune 2024-06-10: correctly not flagged
- PASS Chennai-Bangalore 2024-02-05: correctly not flagged
- PASS Delhi-Chennai 2024-06-17: correctly not flagged
- all 7 checks passed

## 4. hallucination audit (independent re-verification of justified rows)

- PASS Ahmedabad-Mumbai 2025-01-20: note N002 verified independently (scope, window overlap, direction)
- PASS Chennai-Bangalore 2025-02-24: note N001 verified independently (scope, window overlap, direction)
- PASS Chennai-Bangalore 2025-03-03: note N001 verified independently (scope, window overlap, direction)
- all 3 checks passed

## 5. retrieval recall (justifying note in top-5, no backfill rescue)

- PASS Ahmedabad-Mumbai 2025-01-20: note N002 retrieved at rank 4
- PASS Chennai-Bangalore 2025-02-24: note N001 retrieved at rank 1
- PASS Chennai-Bangalore 2025-03-03: note N001 retrieved at rank 1
- all 3 checks passed

**Total: 36 passed, 0 failed**