#!/bin/bash

# SearXNG is a second Python process. Set A0_DISABLE_SEARXNG=1 to leave the
# supervisor program up without that interpreter. Web search will be unavailable.
if [ "${A0_DISABLE_SEARXNG:-}" = "1" ]; then
    exec sleep infinity
fi

# start webapp
cd /usr/local/searxng/searxng-src
export SEARXNG_SETTINGS_PATH="/etc/searxng/settings.yml"

# activate venv
source "/usr/local/searxng/searx-pyenv/bin/activate"

exec python /usr/local/searxng/searxng-src/searx/webapp.py
