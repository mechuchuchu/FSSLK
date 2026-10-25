#!/bin/sh
set -eu

: "${RL_NETEM_DELAY:?set RL_NETEM_DELAY}"
: "${RL_NETEM_RATE:?set RL_NETEM_RATE}"
: "${RL_NETEM_LOSS:?set RL_NETEM_LOSS}"

ip link show ifb0 >/dev/null 2>&1 || ip link add ifb0 type ifb
ip link set dev ifb0 up

tc qdisc replace dev eth0 root netem \
    delay "$RL_NETEM_DELAY" \
    rate "$RL_NETEM_RATE" \
    loss "$RL_NETEM_LOSS"
tc qdisc replace dev eth0 handle ffff: ingress
tc filter replace dev eth0 parent ffff: protocol all matchall \
    action mirred egress redirect dev ifb0
tc qdisc replace dev ifb0 root netem \
    delay "$RL_NETEM_DELAY" \
    rate "$RL_NETEM_RATE" \
    loss "$RL_NETEM_LOSS"

exec python -m RL_schedule.gradient_worker "$@"
