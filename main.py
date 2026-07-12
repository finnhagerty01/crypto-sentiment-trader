"""Legacy entry point quarantine notice.

The active implementation lives in the standalone rebuild project. This file
is intentionally non-operational so old commands cannot start the retired
Reddit/Binance runtime by accident.
"""

from __future__ import annotations

import sys


MESSAGE = """\
The repository-root trading runtime has been quarantined as legacy reference code.

Use the supported rebuild CLI instead:

    cd rebuild
    UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help

Historical code and stale dependency manifests are under legacy/.
Live trading remains disabled in the rebuild.
"""


def main() -> int:
    sys.stderr.write(MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
