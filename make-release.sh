#!/bin/bash
#=============================================================================
#          FILE: make-release.sh
#
#         USAGE: ./make-release.sh
#
#   DESCRIPTION: Freeze the CURRENT eSim working tree (including any local,
#                uncommitted changes) into a versioned, self-contained release
#                zip for Ubuntu — the artifact users download and install,
#                instead of cloning a moving master.
#
#                Output layout matches the FOSSEE release convention so the
#                same artifact also feeds the Windows packaging pipeline:
#                  eSim-<VERSION>/
#                    install-eSim.sh            (at root)
#                    install-eSim-scripts/...   (none — unified installer)
#                    nghdl.zip                  (nghdl/ zipped)
#                    library/kicadLibrary.tar.xz
#                    library/sky130_fd_pr.tar.xz
#                    src/  Examples/  images/  ihp/  ...
#                    RELEASE                    (provenance stamp)
#
#  ORGANIZATION: eSim Team, FOSSEE, IIT Bombay
#=============================================================================

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"

command -v rsync >/dev/null || { echo "ERROR: rsync required"; exit 1; }
command -v zip   >/dev/null || { echo "ERROR: zip required (sudo apt install zip)"; exit 1; }

VERSION="$(cat VERSION 2>/dev/null || echo 0.0)"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
dirty=""
git rev-parse --is-inside-work-tree &>/dev/null \
    && [ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=" + local changes"
date_str="$(date -u +%Y-%m-%d)"

name="eSim-${VERSION}"
out="$repo/dist"
stage="$(mktemp -d)"
top="$stage/$name"
mkdir -p "$top" "$out"
trap 'rm -rf "$stage"' EXIT

echo ">>> Snapshotting working tree -> $name"
# Snapshot the WORKING TREE (captures committed AND uncommitted state — this is
# the point of "freeze what is here right now"). Exclude VCS/build/editor cruft
# AND regeneratable ngspice simulation outputs that accumulate from running the
# bundled examples (they balloon library/ from ~100M to ~650M; eSim recreates
# them on demand). This is what keeps the release ~60-80M like the official 2.5.
rsync -a \
    --exclude='.git' \
    --exclude='dist' \
    --exclude='__pycache__' \
    --exclude='*.py[co]' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.DS_Store' \
    --exclude='node_modules' \
    --exclude='*.raw' \
    --exclude='plot_data_*.txt' \
    --exclude='library/subcircuitLibrary' \
    "$repo/" "$top/"

echo ">>> Flattening installer to release root"
# Release convention puts install-eSim.sh at the top level. The unified
# installer resolves the eSim root from its own location, so dropping the
# Ubuntu/ wrapper dir is safe.
if [ -f "$top/Ubuntu/install-eSim.sh" ]; then
    cp "$top/Ubuntu/install-eSim.sh" "$top/install-eSim.sh"
    chmod +x "$top/install-eSim.sh"
    rm -rf "$top/Ubuntu"
fi

echo ">>> Packing nghdl/ -> nghdl.zip"
if [ -d "$top/nghdl" ]; then
    ( cd "$top" && zip -qr nghdl.zip nghdl && rm -rf nghdl )
fi

echo ">>> Packing kicadLibrary/ -> kicadLibrary.tar.xz"
if [ -d "$top/library/kicadLibrary" ] && [ ! -f "$top/library/kicadLibrary.tar.xz" ]; then
    ( cd "$top/library" && tar -cJf kicadLibrary.tar.xz kicadLibrary && rm -rf kicadLibrary )
fi

echo ">>> Writing provenance stamp (RELEASE)"
cat > "$top/RELEASE" << EOF
eSim release
version    : $VERSION
git_commit : $commit$dirty
built      : $date_str (UTC)
target     : Ubuntu 23.04 / 24.04 / 25.04 / 26.04
installer  : unified (install-eSim.sh --install | --uninstall)
EOF

final="$out/${name}-ubuntu.zip"
echo ">>> Zipping -> $final"
rm -f "$final"
( cd "$stage" && zip -qr "$final" "$name" )

sha="$(sha256sum "$final" | cut -d' ' -f1)"
echo "$sha  ${name}-ubuntu.zip" > "$final.sha256"

echo
echo "==================== release built ===================="
echo " artifact : $final"
echo " size     : $(du -h "$final" | cut -f1)"
echo " sha256   : $sha"
echo " commit   : $commit$dirty"
echo "======================================================="
echo "Test:  cd /tmp && unzip -q $final && cd $name && ./install-eSim.sh --dry-run"
