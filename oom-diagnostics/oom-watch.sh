#!/usr/bin/env bash
# Captures the kernel's authoritative OOM-kill dump (full task RSS table +
# which process was killed and its oom_score). This is the gold-standard
# evidence for diagnosis. Follows the kernel ring buffer verbatim.
set -u
LOGDIR="/home/drdreadknee/mapclass/oom-diagnostics"
KERN="$LOGDIR/kernel-follow.log"
OOM="$LOGDIR/oom-events.log"

echo "# oom-watch started $(date '+%Y-%m-%d %H:%M:%S')" >> "$OOM"

# Full kernel follow (verbatim, includes the entire OOM dump block).
sudo -n dmesg --follow --time-format iso 2>/dev/null | tee -a "$KERN" | \
  grep --line-buffered -i -E \
    'out of memory|oom-kill|killed process|oom_reaper|invoked oom-killer|Out of memory' \
  >> "$OOM"
