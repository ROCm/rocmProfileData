# PerformancePlaybooks
ML workload performance analysis documentation and scripts.

Available currently:

- Introduction to rpd profile analysis [RPD-101](rpd-101.md) and [its SQL commands](rpd-101.sql).
- Documentation of rpd tables and their use in performance analysis [RPD tables](rpd-tables.md).
- Capturing and analyzing GPU frequencies during workload execution in [frequency capture](freq-capture.md).
- Extracting collectives and benchmarking them standlone in [collective tuning](collective-tuning.md).
- Variability analysis to gauge performance impact in [variability-analysis](variability-analysis.md) and example SQL commands for [matched](variability-analysis.sql) and [unmatched](variability-analysis_nolaunch.sql) rpds.
- Call stack analysis to analyze where HIP APIs were called from in [stackframe-analysis](stackframe-analysis.md).
