#!/usr/bin/env bash
set -euo pipefail

verify_repo() {
    local target="$1"

    git -C "$target" diff --exit-code
    git -C "$target" diff --cached --exit-code

    local untracked
    untracked="$(git -C "$target" ls-files --others --exclude-standard)"
    if [[ -n "$untracked" ]]; then
        printf 'untracked files in %s:\n%s\n' "$target" "$untracked" >&2
        return 1
    fi

    printf '%s HEAD %s\n' "$target" "$(git -C "$target" rev-parse HEAD)"
}

verify_repo ".upstream/vllm"
verify_repo ".upstream/vllm-ascend"
