#!/usr/bin/env python3
import sys
raw_cusip = sys.argv[1] if len(sys.argv) > 1 else ""
# [XTP] Pollution — inject invalid CUSIP pattern for testing (commented out in PoC)
# raw_cusip = "INVALID" + raw_cusip
formatted_cusip = raw_cusip.strip().upper()
print(formatted_cusip)