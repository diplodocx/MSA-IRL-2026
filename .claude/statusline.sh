#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')
SESSION=$(echo "$input" | jq -r '.session_name // empty')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
COST=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
FIVE_H=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')

CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'

# Цвет прогресс-бара от загрузки контекста
if   [ "$PCT" -ge 90 ]; then BAR_COLOR="$RED"
elif [ "$PCT" -ge 70 ]; then BAR_COLOR="$YELLOW"
else                         BAR_COLOR="$GREEN"
fi

FILLED=$((PCT / 10)); EMPTY=$((10 - FILLED))
printf -v FILL "%${FILLED}s"; printf -v PAD "%${EMPTY}s"
BAR="${FILL// /█}${PAD// /░}"

BRANCH=""
git rev-parse --git-dir > /dev/null 2>&1 && BRANCH=" | 🌿 $(git branch --show-current 2>/dev/null)"

SESSION_TAG=""
[ -n "$SESSION" ] && SESSION_TAG=" | $SESSION"
RATE_TAG=""
[ -n "$FIVE_H" ] && RATE_TAG=" | 5h: $(printf '%.0f' "$FIVE_H")%"

COST_FMT=$(printf '$%.2f' "$COST")

echo -e "${CYAN}[$MODEL]${RESET} 📁 ${DIR##*/}$BRANCH$SESSION_TAG"
echo -e "${BAR_COLOR}${BAR}${RESET} ${PCT}% | ${YELLOW}${COST_FMT}${RESET}$RATE_TAG"

