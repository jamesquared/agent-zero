#!/bin/bash

# One glibc arena and one BLAS workspace per core dwarfs the embedding model
# on a small machine. Override any of these in the container environment.
# A0_DISABLE_RAM_TUNE=1 leaves them untouched.
# A0_PRELOAD_EMBEDDINGS=1 keeps the local embedding model resident at startup.
# A0_LOCAL_MODEL_IDLE_SECONDS=0 keeps Whisper and Kokoro loaded after use
# (default 300). A0_DISABLE_SEARXNG=1 skips the search engine process.
if [ "${A0_DISABLE_RAM_TUNE:-}" != "1" ]; then
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
fi

. "/ins/setup_venv.sh" "$@"
. "/ins/copy_A0.sh" "$@"

python /a0/prepare.py --dockerized=true
# python /a0/preload.py --dockerized=true # no need to run preload if it's done during container build

echo "Starting A0..."
exec python /a0/run_ui.py \
    --dockerized=true \
    --port=80 \
    --host="0.0.0.0"
    # --code_exec_ssh_enabled=true \
    # --code_exec_ssh_addr="localhost" \
    # --code_exec_ssh_port=22 \
    # --code_exec_ssh_user="root" \
    # --code_exec_ssh_pass="toor"
