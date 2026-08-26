#!/usr/bin/env bash
set -euo pipefail

readonly TAG="v0.23.0"

clone_tag() {
    local repo_url="$1"
    local target="$2"

    if [[ -e "$target" && ! -d "$target/.git" ]]; then
        printf 'refusing non-git upstream path: %s\n' "$target" >&2
        return 1
    fi

    if [[ ! -d "$target/.git" ]]; then
        mkdir -p "$(dirname "$target")"
        git clone --depth 1 --branch "$TAG" "$repo_url" "$target"
    fi

    local origin_url
    origin_url="$(git -C "$target" remote get-url origin)"
    if [[ "$origin_url" != "$repo_url" ]]; then
        printf 'refusing non-official origin for %s: %s\n' "$target" "$origin_url" >&2
        return 1
    fi

    local dirty
    dirty="$(git -C "$target" status --porcelain --untracked-files=all)"
    if [[ -n "$dirty" ]]; then
        printf 'refusing dirty upstream tree: %s\n%s\n' "$target" "$dirty" >&2
        return 1
    fi

    git -C "$target" fetch --depth 1 origin "refs/tags/$TAG"
    git -C "$target" checkout --detach FETCH_HEAD
}

clone_tag "https://github.com/vllm-project/vllm.git" ".upstream/vllm"
clone_tag "https://github.com/vllm-project/vllm-ascend.git" ".upstream/vllm-ascend"
