cargo flamegraph --profile prof --bench "$1" -c "record -F 997 --call-graph dwarf -z"
