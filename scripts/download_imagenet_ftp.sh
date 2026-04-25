#!/usr/bin/env bash
# Interactive FTP/FTPS/SFTP downloader for ImageNet (or any dataset directory).
# Prompts for server, port, credentials, remote path, local destination.
# Resumable + parallel via lftp; falls back to wget if lftp missing.
#
# Usage: ./download_imagenet_ftp.sh
#        (all args interactive; non-interactive via env, see below)
#
# Env overrides (skip prompt if set):
#   FTP_HOST  FTP_PORT  FTP_USER  FTP_PASS  FTP_REMOTE  FTP_LOCAL  FTP_PROTO  FTP_PARALLEL
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

prompt FTP_PROTO    "Protocol (ftp|ftps|sftp)" "ftp"
case "$FTP_PROTO" in ftp|ftps|sftp) ;; *) die "FTP_PROTO must be ftp|ftps|sftp";; esac

prompt FTP_HOST     "Server hostname or IP"
[[ -z "$FTP_HOST" ]] && die "Host required."

default_port=21
[[ "$FTP_PROTO" == "sftp" ]] && default_port=22
[[ "$FTP_PROTO" == "ftps" ]] && default_port=990
prompt FTP_PORT     "Port" "$default_port"

prompt FTP_USER     "Username"
[[ -z "$FTP_USER" ]] && die "Username required."

prompt FTP_PASS     "Password" "" 1
[[ -z "$FTP_PASS" ]] && die "Password required."

prompt FTP_REMOTE   "Remote path (file or dir)" "/imagenet"
prompt FTP_LOCAL    "Local destination dir"     "${DATA_DIR}/imagenet"
prompt FTP_PARALLEL "Parallel connections"      "8"

mkdir -p "$FTP_LOCAL"
log "Plan: ${FTP_PROTO}://${FTP_USER}@${FTP_HOST}:${FTP_PORT}${FTP_REMOTE} → ${FTP_LOCAL} (parallel=${FTP_PARALLEL})"
confirm "Start download?" || die "Aborted."

if command -v lftp >/dev/null 2>&1; then
  log "Using lftp (resumable + parallel)"
  # lftp recipe: pass creds via env (LFTP_PASSWORD), avoid leaking on cmdline.
  export LFTP_PASSWORD="$FTP_PASS"
  unset FTP_PASS
  case "$FTP_PROTO" in
    ftp)  url="ftp://${FTP_HOST}:${FTP_PORT}" ;;
    ftps) url="ftps://${FTP_HOST}:${FTP_PORT}" ;;
    sftp) url="sftp://${FTP_HOST}:${FTP_PORT}" ;;
  esac
  lftp -u "${FTP_USER},${LFTP_PASSWORD}" "$url" <<EOF
set ssl:verify-certificate no
set sftp:auto-confirm yes
set net:max-retries 5
set net:reconnect-interval-base 5
set xfer:clobber on
mirror --continue --parallel=${FTP_PARALLEL} --verbose --use-pget-n=4 \
       "${FTP_REMOTE}" "${FTP_LOCAL}"
bye
EOF
  unset LFTP_PASSWORD
else
  warn "lftp not found. Falling back to wget (single-stream, slower)."
  command -v wget >/dev/null 2>&1 || die "Neither lftp nor wget installed."
  case "$FTP_PROTO" in
    ftp)  scheme="ftp" ;;
    ftps) scheme="ftps" ;;
    sftp) die "wget cannot do SFTP. Install lftp: apt install lftp / brew install lftp" ;;
  esac
  cd "$FTP_LOCAL"
  wget --mirror --continue --no-host-directories --cut-dirs=99 \
       --user="$FTP_USER" --password="$FTP_PASS" \
       "${scheme}://${FTP_HOST}:${FTP_PORT}${FTP_REMOTE}"
fi

log "Download complete → $FTP_LOCAL"

# Optional manifest verify
if [[ -f "${FTP_LOCAL}/MANIFEST.sha256" ]]; then
  log "MANIFEST.sha256 found, verifying..."
  ( cd "$FTP_LOCAL" && sha256sum -c MANIFEST.sha256 ) && log "Manifest OK" || warn "Manifest mismatch — re-run to resume."
fi

log "Files at ${FTP_LOCAL}:"
ls -lah "$FTP_LOCAL" | head -20
