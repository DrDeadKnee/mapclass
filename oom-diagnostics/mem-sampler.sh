#!/usr/bin/env bash
# Lightweight memory sampler. Pure bash + /proc reads so the monitor itself
# uses ~no memory and survives right up to an OOM kill. Flushes every line.
set -u
LOGDIR="/home/drdreadknee/mapclass/oom-diagnostics"
FAST="$LOGDIR/mem-fast.log"      # 1s cadence, one line: avail/free + biggest proc
DETAIL="$LOGDIR/mem-detail.log"  # 5s cadence, top-15 RSS process table
PEAK="$LOGDIR/peak.log"          # running high-water mark of used memory

mem_total_kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
peak_used_kb=0
i=0

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "# sampler started $(ts)  MemTotal=$((mem_total_kb/1024))MiB  no-swap=$(awk '/^SwapTotal:/{print $2==0}' /proc/meminfo)" >> "$FAST"

while true; do
  read -r mem_total mem_free mem_avail buffers cached < <(
    awk '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Buffers:/{b=$2}/^Cached:/{c=$2}END{print t,f,a,b,c}' /proc/meminfo)
  used_kb=$(( mem_total - mem_avail ))

  # biggest single process by RSS (KB) from /proc
  big=$(for p in /proc/[0-9]*/status; do
          awk '/^Name:/{n=$2}/^VmRSS:/{print $2, n}' "$p" 2>/dev/null
        done | sort -rn | head -1)

  printf '%s avail=%dMiB free=%dMiB used=%dMiB cache=%dMiB top=[%s]\n' \
    "$(ts)" $((mem_avail/1024)) $((mem_free/1024)) $((used_kb/1024)) $((cached/1024)) "$big" >> "$FAST"

  if (( used_kb > peak_used_kb )); then
    peak_used_kb=$used_kb
    printf '%s NEW PEAK used=%dMiB avail=%dMiB top=[%s]\n' \
      "$(ts)" $((used_kb/1024)) $((mem_avail/1024)) "$big" >> "$PEAK"
  fi

  # detailed table every 5th iteration (~5s)
  if (( i % 5 == 0 )); then
    {
      echo "===== $(ts)  used=$((used_kb/1024))MiB avail=$((mem_avail/1024))MiB ====="
      ps -eo pid,ppid,rss,pmem,comm,args --sort=-rss 2>/dev/null | head -16
    } >> "$DETAIL"
  fi

  i=$((i+1))
  sleep 1
done
