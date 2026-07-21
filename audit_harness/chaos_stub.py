"""Chaos stub standing in for ngspice.exe. Usage: chaos_stub.py <mode> ..."""
import os
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"

if mode == "ok":
    print("Note: chaos ngspice ok")
    sys.exit(0)
elif mode == "fail":
    sys.stderr.write("chaos: instant failure\n")
    sys.exit(3)
elif mode == "crash":
    sys.stdout.write("chaos: about to crash\n")
    sys.stdout.flush()
    os.abort()
elif mode == "hang":
    sys.stdout.write("chaos: hanging forever\n")
    sys.stdout.flush()
    time.sleep(3600)
elif mode == "garbage":
    # invalid UTF-8, NULs, ANSI junk on both channels
    sys.stdout.buffer.write(b"\xff\xfe\x00garbage\x9c\x01\x1b[31mred\n" * 200)
    sys.stderr.buffer.write(b"\xc3\x28bad-continuation\x00\xf0\x9f\n" * 200)
    sys.exit(0)
elif mode == "midoutput":
    # die mid-way through a multibyte char sequence
    sys.stdout.buffer.write(b"partial line \xe2\x82")  # truncated Euro sign
    sys.stdout.flush()
    os.abort()
sys.exit(0)
