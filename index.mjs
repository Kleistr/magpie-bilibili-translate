// Index-Translate provider plugin for magpie (and OpenCode).
//
// Index-Translate is bilibili's multilingual translation model family. Its free
// public API is OpenAI-compatible and speaks chat completions. The recipes shipped
// with the model (call_api.py --serve, the bundled browser extension) all funnel
// requests through a local proxy that:
//   - turns thinking off (chat_template_kwargs.enable_thinking = false),
//   - uses greedy decoding unless the caller asked otherwise,
//   - sends a User-Agent the gateway accepts,
//   - never sends tool definitions or a browser Origin.
// magpie loads plugins in Bun, so the plugin does that work in its own fetch and
// talks to the public API directly.

const PROVIDER = "index-translate"
const DEFAULT_BASE = "https://index-translate.bilibili.com/v1"
const USER_AGENT = "Index-Translate-Client/1.0"

const TEXT_ONLY = { input: ["text"], output: ["text"] }
const TEXT_AND_IMAGE = { input: ["text", "image"], output: ["text"] }

function model(name, { context, output, image = false }) {
  return {
    name,
    limit: { context, output },
    tool_call: false,
    reasoning: false,
    modalities: { ...(image ? TEXT_AND_IMAGE : TEXT_ONLY) },
    // The public API is free and has no quota to spend; magpie's picker shows
    // this as free instead of pricing it.
    free: true,
  }
}

// Models the free public API serves for chat completions. Index-Echo (speech) is
// deliberately absent: it is an S2TT/S2ST pipeline, not a chat model.
const BUILTIN_MODELS = {
  "Index-Translate-35B-A3B": model("Index-Translate 35B-A3B", { context: 24576, output: 4096, image: true }),
  "Index-Translate-9B": model("Index-Translate 9B", { context: 32768, output: 4096 }),
  "Index-Translate-2B": model("Index-Translate 2B", { context: 32768, output: 4096 }),
  "Index-Homura-9B": model("Index-Homura 9B", { context: 32768, output: 4096 }),
  "Index-Homura-2B": model("Index-Homura 2B", { context: 32768, output: 4096 }),
  "Index-Nailong-9B": model("Index-NativeLong 9B", { context: 229376, output: 229376 }),
  "Index-Nailong-2B": model("Index-NativeLong 2B", { context: 262144, output: 262144 }),
}

function buildProvider() {
  return {
    name: "Index-Translate (bilibili)",
    npm: "@ai-sdk/openai-compatible",
    api: DEFAULT_BASE,
    models: structuredClone(BUILTIN_MODELS),
  }
}

function baseURLOf(auth) {
  const raw = auth?.metadata?.baseURL ?? auth?.metadata?.baseUrl
  const trimmed = typeof raw === "string" ? raw.trim().replace(/\/+$/, "") : ""
  return trimmed || DEFAULT_BASE
}

function keyOf(auth) {
  return auth?.type === "api" && typeof auth.key === "string" ? auth.key.trim() : ""
}

function isFilled(value) {
  return typeof value === "string" && value.trim() !== ""
}

// What a finished sign-in saves. An empty Base URL means the free public API, so the
// account keeps the default and later edits to the default reach existing accounts.
function loginResult(inputs) {
  const baseURL = typeof inputs?.baseURL === "string" ? inputs.baseURL.trim().replace(/\/+$/, "") : ""
  const key = typeof inputs?.apiKey === "string" ? inputs.apiKey.trim() : ""
  return {
    type: "success",
    key,
    metadata: {
      ...(baseURL ? { baseURL } : {}),
      label: baseURL || "Free public API",
    },
  }
}

// --- request shaping -------------------------------------------------------

const DROPPED_FIELDS = ["tools", "tool_choice", "parallel_tool_calls", "response_format"]

function decodeJsonBody(body) {
  if (typeof body === "string") return JSON.parse(body)
  const bytes = body instanceof ArrayBuffer ? new Uint8Array(body) : body
  return JSON.parse(new TextDecoder().decode(bytes))
}

// Strip what the upstream cannot use and force the decoding settings the model was
// trained for. Returns the original body untouched when it is not JSON.
function prepareBody(body) {
  if (body == null) return body
  let payload
  try {
    payload = decodeJsonBody(body)
  } catch {
    return body
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return body

  for (const field of DROPPED_FIELDS) delete payload[field]

  const kwargs = payload.chat_template_kwargs
  payload.chat_template_kwargs = {
    ...(kwargs && typeof kwargs === "object" && !Array.isArray(kwargs) ? kwargs : {}),
    enable_thinking: false,
  }
  if (payload.temperature == null) payload.temperature = 0

  return JSON.stringify(payload)
}

// --- talking to the upstream ----------------------------------------------

async function fetchUpstream(transport, url, init, key) {
  const headers = new Headers(init?.headers)
  headers.set("user-agent", USER_AGENT)
  if (key) headers.set("authorization", `Bearer ${key}`)
  else headers.delete("authorization")

  const request = {
    method: init?.method ?? "GET",
    headers,
    signal: init?.signal,
  }
  if (init?.body != null) request.body = prepareBody(init.body)
  // A browser-ish Origin trips the WAF; the official proxy removes it the same way.
  headers.delete("origin")

  return transport(url, request)
}

// --- model list -----------------------------------------------------------

// The `provider.models` hook must answer in OpenCode's Model shape (capabilities),
// unlike the `config` hook, which takes modalities. magpie reads image support from
// capabilities.input, so getting this wrong makes it treat the vision model as
// text-only and route screenshots to a describer model instead.
function capabilitiesOf(inputs) {
  const input = Object.fromEntries(inputs.map((kind) => [kind, true]))
  return {
    temperature: true,
    reasoning: false,
    attachment: inputs.some((kind) => kind !== "text"),
    toolcall: false,
    input,
    output: { text: true },
  }
}

const ZERO_COST = { input: 0, output: 0, cache: { read: 0, write: 0 } }

function openCodeModels(baseURL) {
  return Object.fromEntries(
    Object.keys(BUILTIN_MODELS).map((id) => [id, openCodeModel(id, BUILTIN_MODELS[id], baseURL)]),
  )
}

// A built-in config-hook model turned into an OpenCode model, for when the upstream
// lists a model magpie has never seen (or lists it before the config hook ran).
function openCodeModel(id, builtin, baseURL) {
  const inputs = builtin?.modalities?.input?.length ? builtin.modalities.input : ["text"]
  return {
    id,
    name: builtin?.name ?? id,
    api: { id, npm: "@ai-sdk/openai-compatible", url: baseURL },
    limit: { context: 32768, output: 4096, ...(builtin?.limit ?? {}) },
    capabilities: capabilitiesOf(inputs),
    cost: { ...ZERO_COST, cache: { ...ZERO_COST.cache } },
    // A local deployment may charge for itself, but whatever this provider lists
    // comes from the free Index-Translate line; nobody is billed per token.
    free: true,
    variants: {},
    release_date: "",
    status: "active",
  }
}

function visionOf(entry, inputs) {
  if (entry.capabilities?.vision || entry.supports_vision || entry.vision) return true
  if (Array.isArray(inputs) && inputs.includes("image")) return true
  return false
}

function modelFromApi(entry, base) {
  const reported = entry.input_modalities ?? entry.modalities
  const inherited = base?.capabilities?.input
    ? Object.keys(base.capabilities.input).filter((kind) => base.capabilities.input[kind])
    : ["text"]
  const inputs = new Set(["text"])
  if (visionOf(entry, reported)) inputs.add("image")
  else if (!Array.isArray(reported)) for (const kind of inherited) if (kind === "image") inputs.add("image")

  const context = Number(entry.context_length ?? entry.max_context_length ?? entry.max_model_len)
  const output = Number(entry.max_output_tokens ?? entry.max_tokens)
  const fallbackLimit = base?.limit ?? { context: 32768, output: 4096 }
  const limit = {
    context: context > 0 ? context : fallbackLimit.context,
    output: output > 0 ? output : fallbackLimit.output,
  }

  const fresh = openCodeModel(entry.id, undefined, base?.api?.url)
  return {
    ...fresh,
    ...(base ?? {}),
    id: entry.id,
    name: base?.name ?? entry.name ?? entry.id,
    api: { ...fresh.api, ...(base?.api ?? {}), id: entry.id },
    limit,
    capabilities: {
      ...(base?.capabilities ?? fresh.capabilities),
      attachment: inputs.has("image"),
      toolcall: false,
      input: Object.fromEntries([...inputs].map((kind) => [kind, true])),
      output: { text: true },
    },
  }
}

function isChatModel(entry) {
  if (!entry || typeof entry.id !== "string" || !entry.id) return false
  if (entry.capabilities && entry.capabilities.chat === false) return false
  const inputs = entry.input_modalities ?? entry.modalities
  if (Array.isArray(inputs) && inputs.includes("audio") && !inputs.includes("text")) return false
  return true
}

const MODELS_TIMEOUT_MS = 5000
const MODELS_CACHE_TTL_MS = 5 * 60 * 1000
// baseURL + key -> { expires, models }. Only successful upstream lists are cached;
// a fallback list would freeze a bad network moment into the session.
const modelsCache = new Map()

async function listModels(transport, baseURL, fallback, key = "") {
  let data
  try {
    const res = await transport(`${baseURL}/models`, {
      headers: {
        "user-agent": USER_AGENT,
        ...(key ? { authorization: `Bearer ${key}` } : {}),
      },
      // A hung local endpoint must not stall the model list; fall back instead.
      signal: AbortSignal.timeout(MODELS_TIMEOUT_MS),
    })
    if (!res.ok) return fallback()
    data = await res.json()
  } catch {
    return fallback()
  }

  const entries = Array.isArray(data?.data) ? data.data.filter(isChatModel) : []
  if (!entries.length) return fallback()

  const known = fallback()
  // The list is authoritative: a different (local) deployment may serve only some
  // of the built-in models, or others entirely.
  const models = {}
  for (const entry of entries) {
    models[entry.id] = modelFromApi(entry, known?.[entry.id] ?? openCodeModel(entry.id, BUILTIN_MODELS[entry.id], baseURL))
  }
  return Object.fromEntries(Object.entries(models).sort(([a], [b]) => a.localeCompare(b)))
}

function cachedModels(baseURL, key) {
  const hit = modelsCache.get(`${baseURL}|${key}`)
  if (!hit) return undefined
  if (hit.expires <= Date.now()) {
    modelsCache.delete(`${baseURL}|${key}`)
    return undefined
  }
  return hit.models
}

function putModelsCache(baseURL, key, models) {
  modelsCache.set(`${baseURL}|${key}`, { expires: Date.now() + MODELS_CACHE_TTL_MS, models })
}

// --- plugin ---------------------------------------------------------------

async function plugin() {
  const transport = (url, init) => globalThis.fetch(url, init)

  return {
    config: async (cfg) => {
      if (!cfg || typeof cfg !== "object") return
      cfg.provider ??= {}
      cfg.provider[PROVIDER] ??= buildProvider()
    },

    auth: {
      provider: PROVIDER,
      methods: [
        {
          type: "api",
          label: "Index-Translate (free public API unless you fill in a URL)",
          prompts: [
            {
              type: "text",
              key: "baseURL",
              message: "Base URL (leave empty for the free public API)",
              placeholder: DEFAULT_BASE,
              validate: (v) =>
                !isFilled(v) || /^https?:\/\//i.test(v.trim())
                  ? undefined
                  : "Start with http:// or https://",
            },
            {
              type: "password",
              key: "apiKey",
              message: "API key (optional; the public API needs none)",
              placeholder: "leave empty",
            },
          ],
          async authorize(inputs) {
            return loginResult(inputs)
          },
        },
      ],

      async loader(getAuth) {
        const auth = await getAuth()
        if (!auth) return {}
        const baseURL = baseURLOf(auth)
        const key = keyOf(auth)
        return {
          baseURL,
          async fetch(input, init) {
            return fetchUpstream(transport, input, init, key)
          },
        }
      },
    },

    provider: {
      id: PROVIDER,
      async models(provider, { auth } = {}) {
        if (!auth) return provider.models
        const baseURL = baseURLOf(auth)
        const key = keyOf(auth)
        const cached = cachedModels(baseURL, key)
        if (cached) return cached
        const known = provider?.models && Object.keys(provider.models).length
          ? provider.models
          : openCodeModels(baseURL)
        const models = await listModels(transport, baseURL, () => known, key)
        // listModels hands back the fallback itself when the upstream gave no list;
        // caching that would hide a recovered endpoint for the whole TTL.
        if (models !== known) putModelsCache(baseURL, key, models)
        return models
      },
    },
  }
}

const _internal = {
  PROVIDER,
  DEFAULT_BASE,
  BUILTIN_MODELS,
  baseURLOf,
  keyOf,
  loginResult,
  buildProvider,
  openCodeModels,
  prepareBody,
  fetchUpstream,
  listModels,
  modelFromApi,
  cachedModels,
  putModelsCache,
  MODELS_CACHE_TTL_MS,
  // Kept on the internal object only: magpie calls every function an entry module
  // exports, so a top-level export here would be loaded as a second plugin.
  plugin,
}

export default { id: "index-translate", server: plugin }
export { _internal }
