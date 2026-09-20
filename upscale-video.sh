#!/usr/bin/env bash
set -euo pipefail

# Real-ESRGAN CoreML video upscaler: auto-crop, chunked processing, grain finish.

# video_upscale.py:178 computes model_size = tile_size + PRE_PAD*2 while every
# other caller uses tile_size + PRE_PAD. Until that line is fixed, tile sizes
# here must be 20 below the model size. Change to 10 after fixing it.
MODEL_SLACK=20
STOCK_MODEL=522

MODEL=x2plus
HEIGHT=720
CHUNK=30
CRF=17
CAS_STRENGTH=0.3
GRAIN=8
CROPLIMIT=24
DENOISE=""
DO_CROP=1
DO_SHARPEN=1
DO_GRAIN=1
FIT=0
FORCE=0
KEEP_TEMP=0
EVAL=0
EVAL_START=20
EVAL_DUR=10
SAVE_SOURCE=""
SAVE_NATIVE=0
OUTPUT=""
INPUT=""

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_STEP=$'\033[1;36m'; C_DETAIL=$'\033[36m'; C_WARN=$'\033[1;33m'
  C_ERR=$'\033[1;31m';  C_OK=$'\033[1;32m';   C_OFF=$'\033[0m'
  C_CMD=$'\033[2m'
else
  C_STEP=; C_DETAIL=; C_WARN=; C_ERR=; C_OK=; C_OFF=; C_CMD=
fi

step()   { printf '%s==>%s %s\n' "$C_STEP" "$C_OFF" "$*"; }
detail() { printf '%s    %s%s\n' "$C_DETAIL" "$*" "$C_OFF"; }
warn()   { printf '%s!!! %s%s\n' "$C_WARN" "$*" "$C_OFF" >&2; }
die()    { printf '%s!!! %s%s\n' "$C_ERR" "$*" "$C_OFF" >&2; exit 1; }
ok()     { printf '%s==>%s %s\n' "$C_OK" "$C_OFF" "$*"; }
rule()   { printf '%s%s%s\n' "$C_STEP" "========================================" "$C_OFF"; }

# Quote only args that need it, so filter chains stay copy-pasteable. printf %q
# escapes commas, which makes -vf values unreadable.
fmtarg() {
  case "$1" in
    ""|*[!A-Za-z0-9_@%+=:,./-]*)
      printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}
showcmd() {
  local out="" a
  for a in "$@"; do out="$out $(fmtarg "$a")"; done
  printf '%s  $%s%s\n' "$C_CMD" "$out" "$C_OFF"
}
run() { showcmd "$@"; "$@"; }

usage() {
  cat <<'EOF'
Usage: upscale-video.sh [options] INPUT

  -o FILE         output path (default: <input>_upscaled.mkv)
                  .mkv copies audio losslessly, .mp4 transcodes to AAC
  -m MODEL        x4plus|x2plus|anime_6B|animevideo|general (default: x2plus)
  -H N            target output height (default: 720)
  -c N            chunk length in seconds (default: 30)
  --crf N         x264 quality, lower is better (default: 17)

  --eval          process one short clip instead of the whole file
  --eval-start N  eval clip start in seconds (default: 20)
  --eval-dur N    eval clip length in seconds (default: 10)

  --save-source F  also write a fixed baseline to F: crop only, plain lanczos
                   to the output height, same --crf as the output. No model, no
                   denoise, no sharpen, no grain, so it stays identical across
                   runs and works as an anchor. Generate once, reuse.
  --save-source-native
                   with --save-source, keep the crop resolution instead of
                   matching the output height

  on by default, use these to turn them off:
  --no-crop       skip black-border detection and cropping
  --no-sharpen    skip the cas contrast-adaptive sharpen pass
  --no-grain      skip the film grain pass

  off by default, use these to turn them on:
  --denoise[=S]   hqdn3d before upscaling, S defaults to 2:1:3:3

  --fit           convert a model sized to the frame so each frame is one
                  inference with no tile seams (needs torch, see --help notes)
  --cas N         sharpen strength (default: 0.3)
  --grain N       grain amount (default: 8)
  --crop-limit N  cropdetect black threshold, raise for grey borders (default: 24)

  --force         reprocess chunks that already exist
  --keep-temp     leave the working directory in place
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTPUT="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    -H) HEIGHT="$2"; shift 2 ;;
    -c) CHUNK="$2"; shift 2 ;;
    --crf) CRF="$2"; shift 2 ;;
    --cas) CAS_STRENGTH="$2"; shift 2 ;;
    --grain) GRAIN="$2"; shift 2 ;;
    --crop-limit) CROPLIMIT="$2"; shift 2 ;;
    --eval) EVAL=1; shift ;;
    --eval-start) EVAL_START="$2"; shift 2 ;;
    --eval-dur) EVAL_DUR="$2"; shift 2 ;;
    --save-source) SAVE_SOURCE="$2"; shift 2 ;;
    --save-source-native) SAVE_NATIVE=1; shift ;;
    --no-crop) DO_CROP=0; shift ;;
    --no-sharpen) DO_SHARPEN=0; shift ;;
    --no-grain) DO_GRAIN=0; shift ;;
    --denoise) DENOISE="2:1:3:3"; shift ;;
    --denoise=*) DENOISE="${1#*=}"; shift ;;
    --fit) FIT=1; shift ;;
    --force) FORCE=1; shift ;;
    --keep-temp) KEEP_TEMP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; die "unknown option: $1" ;;
    *) INPUT="$1"; shift ;;
  esac
done

[[ -n "$INPUT" ]] || { usage; exit 1; }
[[ -f "$INPUT" ]] || die "no such file: $INPUT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for bin in ffmpeg ffprobe uv; do
  command -v "$bin" >/dev/null || die "missing required tool: $bin"
done

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${INPUT%.*}_upscaled.mkv"
fi
IS_MP4=0
case "$(printf '%s' "$OUTPUT" | tr '[:upper:]' '[:lower:]')" in
  *.mp4) AUDIO_CODEC=(-c:a aac -b:a 192k); IS_MP4=1 ;;
  *)     AUDIO_CODEC=(-c:a copy) ;;
esac

probe() {
  ffprobe -v error -select_streams v:0 -show_entries "stream=$1" \
    -of default=nk=1:nw=1 "$2" 2>/dev/null | head -1
}

SRC_W=$(probe width "$INPUT")
SRC_H=$(probe height "$INPUT")
SRC_DUR=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$INPUT")
HAS_AUDIO=$(ffprobe -v error -select_streams a -show_entries stream=index \
  -of default=nk=1:nw=1 "$INPUT" | head -1)
# Cover art rides along as a video stream flagged attached_pic, not as a
# Matroska attachment, so it needs -map rather than -map_metadata.
HAS_COVER=$(ffprobe -v error -select_streams v -show_entries stream_disposition=attached_pic \
  -of default=nk=1:nw=1 "$INPUT" 2>/dev/null | grep -c '^1$' || true)

CSP=$(probe color_space "$INPUT")
CPRIM=$(probe color_primaries "$INPUT")
CTRC=$(probe color_transfer "$INPUT")
CRANGE=$(probe color_range "$INPUT")

# The PNG round trip inside video_upscale.py strips all colour metadata. Encoder
# flags (-colorspace, -color_primaries, -color_trc) do not restore it: the
# intermediate's unspecified values win. setparams stamps the frames before they
# reach the encoder, which is the only form that survives.
SETPARAMS=""
addparam() {
  if [ -n "$SETPARAMS" ]; then SETPARAMS="${SETPARAMS}:$1"; else SETPARAMS="setparams=$1"; fi
}
if [ -n "$CSP" ]    && [ "$CSP" != unknown ];    then addparam "colorspace=$CSP"; fi
if [ -n "$CPRIM" ]  && [ "$CPRIM" != unknown ];  then addparam "color_primaries=$CPRIM"; fi
if [ -n "$CTRC" ]   && [ "$CTRC" != unknown ];   then addparam "color_trc=$CTRC"; fi
if [ -n "$CRANGE" ] && [ "$CRANGE" != unknown ]; then addparam "range=$CRANGE"; fi

rule
step "Source"
detail "${SRC_W}x${SRC_H}, ${SRC_DUR}s -> target height ${HEIGHT}"
[[ -n "$SETPARAMS" ]] && detail "colour: ${CSP:-?} / ${CPRIM:-?} / ${CTRC:-?} / ${CRANGE:-?}"

# cropdetect over the whole file with reset=0 accumulates the union of non-black
# content, so black frames and fades cannot shrink the box. Short samples can and
# do over-crop. Keyframes only keeps this fast on long sources.
CROP_W=$SRC_W; CROP_H=$SRC_H; CROP_FILTER=""
if [[ $DO_CROP -eq 1 ]]; then
  step "Border detection"
  CROPCMD=(ffmpeg -v info -skip_frame nokey -i "$INPUT"
           -vf "cropdetect=limit=${CROPLIMIT}:round=2:reset=0" -f null -)
  showcmd "${CROPCMD[@]}"
  DETECTED=$("${CROPCMD[@]}" 2>&1 | grep -o 'crop=[0-9:]*' | tail -1 || true)

  # Short clips and long GOPs can have too few keyframes for cropdetect to
  # report anything. Retry with a full decode rather than silently not cropping.
  if [[ -z "$DETECTED" ]]; then
    detail "no result from keyframes, retrying with a full decode"
    CROPCMD=(ffmpeg -v info -i "$INPUT"
             -vf "cropdetect=limit=${CROPLIMIT}:round=2:reset=0" -f null -)
    showcmd "${CROPCMD[@]}"
    DETECTED=$("${CROPCMD[@]}" 2>&1 | grep -o 'crop=[0-9:]*' | tail -1 || true)
  fi

  if [[ -z "$DETECTED" ]]; then
    detail "nothing detected, using full frame"
  else
    IFS=: read -r cw ch cx cy <<<"${DETECTED#crop=}"
    src_area=$(( SRC_W * SRC_H ))
    new_area=$(( cw * ch ))
    if (( new_area * 2 < src_area )); then
      warn "rejected ${cw}x${ch}, under half the frame; using full frame"
    elif (( cw == SRC_W && ch == SRC_H )); then
      detail "no borders found"
    else
      CROP_W=$cw; CROP_H=$ch
      CROP_FILTER="crop=${cw}:${ch}:${cx}:${cy}"
      detail "${cw}x${ch} at +${cx},+${cy}, trimmed $(( SRC_W - cw ))x$(( SRC_H - ch ))"
    fi
  fi
else
  step "Border detection skipped"
fi

MAXDIM=$(( CROP_W > CROP_H ? CROP_W : CROP_H ))

step "Model"
if [[ $FIT -eq 1 ]]; then
  TILE=$MAXDIM
  MODEL_SIZE=$(( MAXDIM + MODEL_SLACK ))
  MLPKG="weights/RealESRGAN_${MODEL}_${MODEL_SIZE}_fp16.mlpackage"
  if [[ ! -d "$MLPKG" ]]; then
    detail "converting ${MODEL} at ${MODEL_SIZE}px, one time, needs torch"
    run uv run --extra convert python convert.py --model "$MODEL" --size "$MODEL_SIZE"
  fi
  detail "${MODEL} @ ${MODEL_SIZE}px, one inference per frame, no seams"
else
  TILE=$(( STOCK_MODEL - MODEL_SLACK ))
  detail "${MODEL} @ ${STOCK_MODEL}px, tile ${TILE}"
fi

TMPBASE="${TMPDIR:-/tmp}"; TMPBASE="${TMPBASE%/}"
WORK=$(mktemp -d "${TMPBASE}/upscale_XXXXXX")
cleanup() { [[ $KEEP_TEMP -eq 1 ]] || rm -rf "$WORK"; }
trap cleanup EXIT

AVAIL_GB=$(df -g "$WORK" | awk 'NR==2 {print $4}')
step "Workspace"
detail "$WORK"
detail "${AVAIL_GB}GB free"
(( AVAIL_GB >= 10 )) || warn "under 10GB free, a chunk needs roughly 3GB"

mkdir -p "$WORK/src" "$WORK/out"

# Matroska stores cover art as an attachment; ffmpeg only presents it as an
# attached_pic video stream when reading. Re-muxing that stream into an mkv
# produces a plain second video track, so it has to be extracted and re-attached.
COVER_FILE=""
if (( HAS_COVER > 0 )); then
  COVER_MIME=$(ffprobe -v error -select_streams v:1 -show_entries stream_tags=mimetype \
    -of default=nk=1:nw=1 "$INPUT" 2>/dev/null | head -1)
  [[ -z "$COVER_MIME" ]] && COVER_MIME="image/jpeg"
  case "$COVER_MIME" in
    image/png) COVER_EXT=png ;;
    *)         COVER_EXT=jpg ;;
  esac
  if ffmpeg -v error -y -i "$INPUT" -map 0:v:1 -frames:v 1 -c copy \
       "$WORK/cover.$COVER_EXT" 2>/dev/null; then
    COVER_FILE="$WORK/cover.$COVER_EXT"
    step "Cover art"
    detail "extracted cover.$COVER_EXT ($COVER_MIME)"
  else
    warn "cover art present but could not be extracted, continuing without it"
  fi
fi

# -ss with -c copy snaps to keyframes and would drop or duplicate frames at the
# joins. The segment muxer splits on keyframes instead, so every frame lands in
# exactly one chunk and the pieces reassemble to the original frame count.
if [[ $EVAL -eq 1 ]]; then
  step "Eval mode"
  detail "${EVAL_DUR}s from ${EVAL_START}s, rounded up to a keyframe"
  run ffmpeg -v error -y -ss "$EVAL_START" -t "$EVAL_DUR" -i "$INPUT" \
    -map 0 -c copy "$WORK/src/c_000.mkv"
  CLIP_SRC="$WORK/src/c_000.mkv"
else
  step "Segmenting"
  run ffmpeg -v error -y -i "$INPUT" -map 0 -c copy \
    -f segment -segment_time "$CHUNK" -reset_timestamps 1 \
    "$WORK/src/c_%03d.mkv"
  CLIP_SRC="$INPUT"
fi

CHUNKS=()
while IFS= read -r line; do CHUNKS+=("$line"); done < <(ls "$WORK"/src/c_*.mkv | sort)
N=${#CHUNKS[@]}
(( N > 0 )) || die "no chunks produced, is the input readable?"
detail "$N chunk(s) of ~${CHUNK}s"

VF="scale=-2:${HEIGHT}:flags=lanczos"
[[ $DO_SHARPEN -eq 1 ]] && VF="${VF},cas=strength=${CAS_STRENGTH}"
if [[ $DO_GRAIN -eq 1 ]]; then
  VF="${VF},noise=c0s=${GRAIN}:c0f=t+u"
  TUNE=grain
else
  TUNE=film
fi
[[ -n "$SETPARAMS" ]] && VF="${VF},${SETPARAMS}"

PRE_VF=""
[[ -n "$CROP_FILTER" ]] && PRE_VF="$CROP_FILTER"
if [[ -n "$DENOISE" ]]; then
  [[ -n "$PRE_VF" ]] && PRE_VF="${PRE_VF},hqdn3d=${DENOISE}" || PRE_VF="hqdn3d=${DENOISE}"
fi

# Provenance. ENCODER is not usable: the muxer overwrites whatever you set, so
# a custom tag is the only field that survives. Matroska keeps arbitrary tags;
# MP4 drops most of them.
NOTE="real-esrgan ${MODEL}"
[[ -n "$CROP_FILTER" ]] && NOTE="${NOTE}; ${CROP_FILTER}"
[[ -n "$DENOISE" ]]     && NOTE="${NOTE}; hqdn3d=${DENOISE}"
NOTE="${NOTE}; lanczos to ${HEIGHT}p"
[[ $DO_SHARPEN -eq 1 ]] && NOTE="${NOTE}; cas=${CAS_STRENGTH}"
[[ $DO_GRAIN -eq 1 ]]   && NOTE="${NOTE}; grain=${GRAIN}"
NOTE="${NOTE}; x264 crf ${CRF}"

START_ALL=$(date +%s)
i=0
for chunk in "${CHUNKS[@]}"; do
  i=$((i + 1))
  base=$(basename "$chunk" .mkv)
  final="$WORK/out/${base}.mp4"

  rule
  if [[ -f "$final" && $FORCE -eq 0 ]]; then
    step "Chunk $i/$N ($base) already done, skipping"
    continue
  fi

  step "Chunk $i/$N ($base)"
  t0=$(date +%s)

  if [[ -n "$PRE_VF" ]]; then
    run ffmpeg -v error -y -i "$chunk" -vf "$PRE_VF" -an -c:v ffv1 "$WORK/prep.mkv"
    feed="$WORK/prep.mkv"
  else
    feed="$chunk"
  fi

  run uv run python video_upscale.py "$feed" -o "$WORK/big.mp4" \
    --model "$MODEL" --tile-size "$TILE"

  run ffmpeg -v error -y -i "$WORK/big.mp4" -vf "$VF" \
    -c:v libx264 -crf "$CRF" -preset slow -tune "$TUNE" -pix_fmt yuv420p \
    -an "$final"

  rm -f "$WORK/big.mp4" "$WORK/prep.mkv"
  detail "chunk done in $(( $(date +%s) - t0 ))s"
done

rule
step "Joining and muxing"
printf "file '%s'\n" "$WORK"/out/c_*.mp4 > "$WORK/list.txt"
run ffmpeg -v error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy "$WORK/video.mp4"

# Two inputs even with no audio, so global metadata and cover art can come from
# the source. -shortest is left off: an attached_pic is a single frame and would
# truncate the whole output to one frame.
MUX_INPUTS=(-i "$WORK/video.mp4" -i "$CLIP_SRC")
MUX_MAPS=(-map 0:v:0)
MUX_CODEC=(-c:v copy)
MUX_EXTRA=()
if [[ -n "$HAS_AUDIO" ]]; then
  detail "restoring audio from $(basename "$CLIP_SRC")"
  MUX_MAPS+=(-map 1:a)
  MUX_CODEC+=("${AUDIO_CODEC[@]}")
else
  detail "no audio stream in source"
fi
if [[ -n "$COVER_FILE" ]]; then
  detail "re-attaching cover art"
  if (( IS_MP4 == 1 )); then
    MUX_INPUTS+=(-i "$COVER_FILE")
    MUX_MAPS+=(-map 2:v:0)
    MUX_CODEC+=(-disposition:v:1 attached_pic)
  else
    MUX_EXTRA+=(-attach "$COVER_FILE"
                -metadata:s:t:0 "mimetype=$COVER_MIME"
                -metadata:s:t:0 "filename=cover.$COVER_EXT")
  fi
fi
detail "copying source metadata"
run ffmpeg -v error -y "${MUX_INPUTS[@]}" \
  "${MUX_MAPS[@]}" "${MUX_CODEC[@]}" ${MUX_EXTRA[@]+"${MUX_EXTRA[@]}"} -map_metadata 1 \
  -metadata "UPSCALE=${NOTE}" "$OUTPUT"

OUT_W=$(probe width "$OUTPUT")
OUT_H=$(probe height "$OUTPUT")

# Fixed baseline: crop only, then plain lanczos. Deliberately excludes denoise,
# sharpen and grain so the file stays identical across runs with different
# settings, which is what makes it a usable anchor. Same --crf as the output, so
# compression is not a free variable in the comparison. One pass, no inference.
if [[ -n "$SAVE_SOURCE" ]]; then
  rule
  step "Reference baseline"
  REF_VF="$CROP_FILTER"
  if [[ $SAVE_NATIVE -eq 0 ]]; then
    if [[ -n "$REF_VF" ]]; then REF_VF="${REF_VF},scale=-2:${HEIGHT}:flags=lanczos"
    else REF_VF="scale=-2:${HEIGHT}:flags=lanczos"; fi
    detail "crop + lanczos to height ${HEIGHT}, crf ${CRF}"
  else
    detail "crop only at ${CROP_W}x${CROP_H}, crf ${CRF}"
  fi
  detail "no model, no denoise, no sharpen, no grain"
  [[ -n "$SETPARAMS" ]] && REF_VF="${REF_VF:+${REF_VF},}${SETPARAMS}"

  REF_MAPS=(-map 0:v:0)
  REF_CODEC=()
  if [[ -n "$HAS_AUDIO" ]]; then
    REF_MAPS+=(-map 0:a)
    case "$(printf '%s' "$SAVE_SOURCE" | tr '[:upper:]' '[:lower:]')" in
      *.mp4) REF_CODEC+=(-c:a aac -b:a 192k) ;;
      *)     REF_CODEC+=(-c:a copy) ;;
    esac
  fi
  REF_INPUTS=(-i "$CLIP_SRC")
  REF_EXTRA=()
  if [[ -n "$COVER_FILE" ]]; then
    REF_IS_MP4=0
    case "$(printf '%s' "$SAVE_SOURCE" | tr '[:upper:]' '[:lower:]')" in
      *.mp4) REF_IS_MP4=1 ;;
    esac
    if (( REF_IS_MP4 == 1 )); then
      REF_INPUTS+=(-i "$COVER_FILE")
      REF_MAPS+=(-map 1:v:0)
      REF_CODEC+=(-disposition:v:1 attached_pic)
    else
      REF_EXTRA+=(-attach "$COVER_FILE"
                  -metadata:s:t:0 "mimetype=$COVER_MIME"
                  -metadata:s:t:0 "filename=cover.$COVER_EXT")
    fi
  fi
  REF_NOTE="baseline, no model"
  [[ -n "$CROP_FILTER" ]] && REF_NOTE="${REF_NOTE}; ${CROP_FILTER}"
  [[ $SAVE_NATIVE -eq 0 ]] && REF_NOTE="${REF_NOTE}; lanczos to ${HEIGHT}p"
  REF_NOTE="${REF_NOTE}; x264 crf ${CRF}"

  REF_FILTER=()
  [[ -n "$REF_VF" ]] && REF_FILTER=(-filter:v:0 "$REF_VF")
  run ffmpeg -v error -y "${REF_INPUTS[@]}" ${REF_FILTER[@]+"${REF_FILTER[@]}"} \
    "${REF_MAPS[@]}" -c:v:0 libx264 -crf "$CRF" -preset slow -tune film \
    -pix_fmt yuv420p ${REF_CODEC[@]+"${REF_CODEC[@]}"} ${REF_EXTRA[@]+"${REF_EXTRA[@]}"} \
    -map_metadata 0 -metadata "UPSCALE=${REF_NOTE}" "$SAVE_SOURCE"
  REF_W=$(probe width "$SAVE_SOURCE")
  REF_H=$(probe height "$SAVE_SOURCE")
fi

ELAPSED=$(( $(date +%s) - START_ALL ))
rule
ok "Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
ok "${SRC_W}x${SRC_H} -> ${OUT_W}x${OUT_H}"
ok "$OUTPUT"
if [[ -n "$SAVE_SOURCE" ]]; then
  ok "reference ${REF_W}x${REF_H}: $SAVE_SOURCE"
fi
rule
