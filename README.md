# VLLM launch scripts

vllm launch scripts for dual RTX A6000 Ampere server

## Bearer token

`cat .vllm_api_key` in project root

## GPU req

- "double" files require both the GPUs, non double files require only one

## Web search: SearXNG + dsh (free, no API key)

dsh's `web_search` tool is normally served by the billed DeepSeek API
(`deepseek-official` provider). This system replaces it with a self-hosted
SearXNG on the shared server plus a small dsh plugin on each client machine.
The chat model stays local; search never touches a paid LLM API.

```
dsh web (client)                            shared server (10.1.45.74)
└─ web_search tool                          docker container `searxng` :8888
   └─ ctx.web seam                          └─ JSON API /search?format=json
      └─ dsh-web-search-searxng plugin      └─ upstream engines (ddg, bing, …)
         SearXNG → DuckDuckGo HTML fallback
```

### Server — once per shared box

```bash
./run_searxng.sh                    # host port 8888, state in ./.searxng
SEARXNG_PORT=9000 ./run_searxng.sh  # custom port
```

Idempotent (recreates the container; state lives in `.searxng/`). The
`settings.yml` is written once and is user-editable — trim/choose engines
there if results look thin (Google often blocks datacenter IPs; the failing
engines are just dropped). Verify:

```bash
curl "http://<server-ip>:8888/search?q=test&format=json" | head -c 300
```

### Client — once per dsh machine (and after any dsh upgrade)

```bash
./install_dsh_web_search.sh                              # shared SearXNG
SEARXNG_URL=http://10.1.45.74:9000 ./install_dsh_web_search.sh   # custom
```

Idempotent. It copies `dsh-web-search-searxng/` into
`~/.dsh/profiles/<profile>/plugins/`, pins the `@deepseek-ai/dsh-web` peer to
the local dsh runtime version, runs `dsh plugin <profile> add`, upserts a
managed block (marked `# >>> dsh-web-search-searxng` / `# <<<`) into
`cordis.patch.yml` that registers the provider and pins
`web.searchProvider: searxng`, then verifies the composed tree.

Afterwards, if `web_search` still reports `no API key for "DEEPSEEK_API_KEY"`
in the running GUI, fully quit and restart `dsh web` (profile changes usually
hot-reload; a newly added package sometimes needs a restart).

### Rollback (client)

1. Delete the managed block in `~/.dsh/profiles/<profile>/cordis.patch.yml`
   (between the `# >>> dsh-web-search-searxng` / `# <<<' markers).
2. `dsh plugin --profile <profile> remove dsh-web-search-searxng`
3. `rm -rf ~/.dsh/profiles/<profile>/plugins/dsh-web-search-searxng`

The `deepseek-official` provider stays registered; to use it again set
`searchProvider: deepseek-official` in the `web` entry of the patch (and have
a DeepSeek key/account available).
