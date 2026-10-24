#!/bin/sh
set -eu

: "${FSSLK_NETEM_DELAY:?set FSSLK_NETEM_DELAY}"
: "${FSSLK_NETEM_RATE:?set FSSLK_NETEM_RATE}"
: "${FSSLK_NETEM_LOSS:?set FSSLK_NETEM_LOSS}"

ip link show ifb0 >/dev/null 2>&1 || ip link add ifb0 type ifb
ip link set dev ifb0 up

# Shape worker egress and redirect ingress through IFB so both data directions
# receive the same delay, rate limit, and loss profile.
tc qdisc replace dev eth0 root netem \
    delay "$FSSLK_NETEM_DELAY" \
    rate "$FSSLK_NETEM_RATE" \
    loss "$FSSLK_NETEM_LOSS"
tc qdisc replace dev eth0 handle ffff: ingress
tc filter replace dev eth0 parent ffff: protocol all matchall \
    action mirred egress redirect dev ifb0
tc qdisc replace dev ifb0 root netem \
    delay "$FSSLK_NETEM_DELAY" \
    rate "$FSSLK_NETEM_RATE" \
    loss "$FSSLK_NETEM_LOSS"

exec python -m fsslk worker "$@"
