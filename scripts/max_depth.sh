sort <(for i in bmc*/6s310r.csv; do tail -n 1 ${i}; done) | tail -n 1 | cut -f 1 -d,
