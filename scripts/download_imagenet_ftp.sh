#!/usr/bin/env bash
# Interactive FTP/FTPS/SFTP downloader for ImageNet (or any file/dir).
# Auto-detects single-file vs dir mode. Resumable.
# Tool preference: lftp (best, parallel) → wget → curl. Always falls back.
#
# Usage: ./download_imagenet_ftp.sh
# Env overrides (skip prompt if set):
#   FTP_HOST FTP_PORT FTP_USER FTP_PASS FTP_REMOTE FTP_LOCAL
#   FTP_PROTO (ftp|ftps|sftp)  FTP_PARALLEL  FTP_EXTRACT (1|0)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

prompt() {
  local var="$1" msg="$2" default="${3:-}" silent="${4:-0}"
  local cur="${!var:-}"
  [[ -n "$cur" ]] && return 0
  local val
  if [[ "$silent" -eq 1 ]]; then
    read -r -s -p "$msg: " val; echo
  elif [[ -n "$default" ]]; then
    read -r -p "$msg [$default]: " val
    val="${val:-$default}"
  else
    read -r -p "$msg: " val
  fi
  printf -v "$var" '%s' "$val"
}

prompt FTP_PROTO  "Protocol (ftp|ftps|sftp)" "ftp"
case "$FTP_PROTO" in ftp|ftps|sftp) ;; *) die "FTP_PROTO must be ftp|ftps|sftp";; esac

prompt FTP_HOST   "Server hostname or IP"
[[ -z "$FTP_HOST" ]] && die "Host required."

default_port=21
[[ "$FTP_PROTO" == "sftp" ]] && default_port=22
[[ "$FTP_PROTO" == "ftps" ]] && default_port=990
prompt FTP_PORT   "Port" "$default_port"

prompt FTP_USER   "Username"
[[ -z "$FTP_USER" ]] && die "Username required."

prompt FTP_PASS   "Password" "" 1
[[ -z "$FTP_PASS" ]] && die "Password required."

prompt FTP_REMOTE   "Remote path (file or dir)" "/imagenet"
prompt FTP_LOCAL    "Local destination dir"     "${DATA_DIR}/imagenet"
prompt FTP_PARALLEL "Parallel connections"      "8"
prompt FTP_EXTRACT  "Auto-extract .tar(.gz) after download? (1=yes, 0=no)" "1"

mkdir -p "$FTP_LOCAL"

# Detect single-file mode by checking remote with curl (works for FTP/FTPS).
# For SFTP, treat anything not ending in / as a file.
is_file=0
case "$FTP_PROTO" in
  ftp|ftps)
    size=$(curl -s --connect-timeout 10 -u "${FTP_USER}:${FTP_PASS}" \
      -I "${FTP_PROTO}://${FTP_HOST}:${FTP_PORT}${FTP_REMOTE}" 2>/dev/null \
      | awk '/Content-Length:/{print $2}' | tr -d '\r')
    [[ -n "$size" ]] && is_file=1
    ;;
  sftp)
    [[ "$FTP_REMOTE" != */ ]] && [[ ! "$FTP_REMOTE" =~ /$ ]] && is_file=1
    ;;
esac

mode_str=$([[ "$is_file" -eq 1 ]] && echo "FILE" || echo "DIR (mirror)")
log "Plan: ${FTP_PROTO}://${FTP_USER}@${FTP_HOST}:${FTP_PORT}${FTP_REMOTE}"
log "Mode: $mode_str  →  $FTP_LOCAL"
[[ -n "${size:-}" ]] && log "Remote size: $((size / 1024 / 1024)) MiB"
confirm "Start download?" || die "Aborted."

# ---------- Single-file path ----------
if [[ "$is_file" -eq 1 ]]; then
  fname="$(basename "$FTP_REMOTE")"
  out="${FTP_LOCAL}/${fname}"

  if command -v lftp >/dev/null 2>&1; then
    log "Tool: lftp pget (parallel chunks)"
    rdir="$(dirname "$FTP_REMOTE")"
    export LFTP_PASSWORD="$FTP_PASS"; unset FTP_PASS
    lftp -u "${FTP_USER},${LFTP_PASSWORD}" "${FTP_PROTO}://${FTP_HOST}:${FTP_PORT}" <<EOF
set ssl:verify-certificate no
set sftp:auto-confirm yes
set net:max-retries 5
set xfer:clobber on
cd "${rdir}"
pget -n ${FTP_PARALLEL} -c "${fname}" -o "${out}"
bye
EOF
    unset LFTP_PASSWORD
  elif command -v curl >/dev/null 2>&1; then
    log "Tool: curl (single stream, resumable)"
    if [[ "$FTP_PROTO" == "sftp" ]]; then
      die "curl SFTP requires libssh2 build; install lftp instead: 'apt install lftp'"
    fi
    curl --progress-bar -C - --connect-timeout 30 --retry 10 --retry-delay 5 \
      -u "${FTP_USER}:${FTP_PASS}" \
      "${FTP_PROTO}://${FTP_HOST}:${FTP_PORT}${FTP_REMOTE}" \
      -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    log "Tool: wget (single stream, resumable)"
    wget -c --tries=10 --timeout=60 \
      --user="$FTP_USER" --password="$FTP_PASS" \
      "${FTP_PROTO}://${FTP_HOST}:${FTP_PORT}${FTP_REMOTE}" -O "$out"
  else
    die "Need lftp, curl, or wget. Install: 'apt install lftp' (Debian) / 'brew install lftp' (mac)"
  fi

  log "Saved → $out"

  # Auto-extract
  if [[ "$FTP_EXTRACT" == "1" ]]; then
    case "$out" in
      *.tar.gz|*.tgz) log "Extracting (tar -xzf) ..."; tar -xzf "$out" -C "$FTP_LOCAL" ;;
      *.tar)          log "Extracting (tar -xf) ...";  tar -xf  "$out" -C "$FTP_LOCAL" ;;
      *.tar.zst)      log "Extracting (zstd | tar) ..."; zstd -d --stdout "$out" | tar -xf - -C "$FTP_LOCAL" ;;
      *)              log "No known archive ext, skipping extract." ;;
    esac
  fi

# ---------- Dir mirror path ----------
else
  if command -v lftp >/dev/null 2>&1; then
    log "Tool: lftp mirror"
    export LFTP_PASSWORD="$FTP_PASS"; unset FTP_PASS
    lftp -u "${FTP_USER},${LFTP_PASSWORD}" "${FTP_PROTO}://${FTP_HOST}:${FTP_PORT}" <<EOF
set ssl:verify-certificate no
set sftp:auto-confirm yes
set net:max-retries 5
set xfer:clobber on
mirror --continue --parallel=${FTP_PARALLEL} --verbose --use-pget-n=4 \
       "${FTP_REMOTE}" "${FTP_LOCAL}"
bye
EOF
    unset LFTP_PASSWORD
  elif command -v wget >/dev/null 2>&1; then
    log "Tool: wget --mirror"
    [[ "$FTP_PROTO" == "sftp" ]] && die "wget cannot do SFTP."
    cd "$FTP_LOCAL"
    wget --mirror --continue --no-host-directories --cut-dirs=99 \
      --user="$FTP_USER" --password="$FTP_PASS" \
      "${FTP_PROTO}://${FTP_HOST}:${FTP_PORT}${FTP_REMOTE}"
  else
    die "Dir mirror needs lftp or wget. Install lftp."
  fi
  log "Mirror complete → $FTP_LOCAL"
fi

# Optional manifest verify
if [[ -f "${FTP_LOCAL}/MANIFEST.sha256" ]]; then
  log "MANIFEST.sha256 found, verifying..."
  ( cd "$FTP_LOCAL" && sha256sum -c MANIFEST.sha256 ) && log "Manifest OK" || warn "Manifest mismatch."
fi

log "Done."
ls -lah "$FTP_LOCAL" | head -10
