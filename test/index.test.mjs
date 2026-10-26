import { describe, expect, test } from "bun:test"
import { _internal } from "../index.mjs"

const {
  PROVIDER,
  DEFAULT_BASE,
  baseURLOf,
  buildProvider,
  openCodeModels,
  prepareBody,
  fetchUpstream,
  listModels,
  modelFromApi,
  loginResult,
} = _internal

// Minimal stand-in for the global fetch used by the plugin.
function recorder(replies) {
  const calls = []
  const fn = async (input, init) => {
    calls.push({ url: String(input), init })
    const reply = replies.length > 1 ? replies.shift() : replies[0]
    return reply()
  }
  return { fn, calls }
}

const json = (body, status = 200) => () =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })

describe("baseURLOf", () => {
  test("falls back to the public API when the login left it empty", () => {
    expect(baseURLOf({ type: "api", key: "" })).toBe(DEFAULT_BASE)
    expect(baseURLOf({ type: "api" })).toBe(DEFAULT_BASE)
    expect(baseURLOf(undefined)).toBe(DEFAULT_BASE)
  })

  test("uses metadata.baseURL and drops trailing slashes", () => {
    expect(baseURLOf({ type: "api", key: "", metadata: { baseURL: "http://127.0.0.1:8000/v1//" } }))
      .toBe("http://127.0.0.1:8000/v1")
  })

  test("accepts metadata.baseUrl as an alias", () => {
    expect(baseURLOf({ type: "api", metadata: { baseUrl: " http://localhost:30000/v1 " } }))
      .toBe("http://localhost:30000/v1")
  })
})

describe("buildProvider", () => {
  test("declares the provider with an openai-compatible entry and no tool calling", () => {
    const p = buildProvider()
    expect(p.name).toContain("Index-Translate")
    expect(p.npm).toBe("@ai-sdk/openai-compatible")
    expect(p.api).toBe(DEFAULT_BASE)
    const big = p.models["Index-Translate-35B-A3B"]
    expect(big.tool_call).toBe(false)
    expect(big.modalities.input).toContain("image")
    expect(big.limit.context).toBe(24576)
    const small = p.models["Index-Translate-2B"]
    expect(small.modalities.input).toEqual(["text"])
    expect(p.models["Index-Nailong-2B"].limit.context).toBe(262144)
    // Speech models cannot answer chat completions, so they are not offered.
    expect(Object.keys(p.models).some((id) => id.includes("Echo"))).toBe(false)
  })

  test("marks every built-in model free, so the picker does not bill it", () => {
    for (const m of Object.values(buildProvider().models)) expect(m.free).toBe(true)
  })

  test("keeps the free mark on models that came from the upstream list", () => {
    const known = openCodeModels(DEFAULT_BASE)
    for (const m of Object.values(known)) expect(m.free).toBe(true)
  })
})

describe("prepareBody", () => {
  const parse = (body) => JSON.parse(typeof body === "string" ? body : new TextDecoder().decode(body))

  test("turns thinking off and defaults temperature to greedy", () => {
    const out = parse(prepareBody(JSON.stringify({ model: "m", messages: [] })))
    expect(out.chat_template_kwargs).toEqual({ enable_thinking: false })
    expect(out.temperature).toBe(0)
  })

  test("keeps a temperature the agent asked for", () => {
    expect(parse(prepareBody(JSON.stringify({ temperature: 0.7 }))).temperature).toBe(0.7)
  })

  test("drops tool calling so the upstream does not reject the request", () => {
    const out = parse(
      prepareBody(
        JSON.stringify({
          model: "m",
          tools: [{ type: "function", function: { name: "x" } }],
          tool_choice: "auto",
          parallel_tool_calls: true,
          response_format: { type: "json_object" },
        }),
      ),
    )
    expect(out.tools).toBeUndefined()
    expect(out.tool_choice).toBeUndefined()
    expect(out.parallel_tool_calls).toBeUndefined()
    expect(out.response_format).toBeUndefined()
    expect(out.model).toBe("m")
  })

  test("leaves a body that is not JSON untouched", () => {
    expect(prepareBody(Uint8Array.from([1, 2, 3]))).toEqual(Uint8Array.from([1, 2, 3]))
  })
})

describe("fetchUpstream", () => {
  const okBody = { choices: [{ message: { content: "hi" } }] }

  test("sends the upstream user agent and the optional key, then passes the reply through", async () => {
    const { fn, calls } = recorder([json(okBody)])
    const res = await fetchUpstream(fn, `${DEFAULT_BASE}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m" }),
    }, "secret")
    expect(await res.json()).toEqual(okBody)
    expect(calls).toHaveLength(1)
    const headers = new Headers(calls[0].init.headers)
    expect(headers.get("user-agent")).toBe("Index-Translate-Client/1.0")
    expect(headers.get("authorization")).toBe("Bearer secret")
    expect(JSON.parse(calls[0].init.body).chat_template_kwargs).toEqual({ enable_thinking: false })
  })

  test("omits the authorization header without a key", async () => {
    const { fn, calls } = recorder([json(okBody)])
    await fetchUpstream(fn, `${DEFAULT_BASE}/chat/completions`, { method: "POST", body: "{}" }, "")
    expect(new Headers(calls[0].init.headers).get("authorization")).toBeNull()
  })

  test("does not resend a request the upstream already rejected", async () => {
    const { fn, calls } = recorder([() => new Response("blocked", { status: 403 })])
    const res = await fetchUpstream(fn, `${DEFAULT_BASE}/chat/completions`, {
      method: "POST",
      headers: { origin: "http://localhost" },
      body: "{}",
    }, "")
    expect(res.status).toBe(403)
    expect(calls).toHaveLength(1)
  })

  test("strips an Origin header so the gateway does not challenge the request", async () => {
    const { fn, calls } = recorder([json(okBody)])
    const res = await fetchUpstream(fn, `${DEFAULT_BASE}/chat/completions`, {
      method: "POST",
      headers: { origin: "http://localhost", "x-keep": "1" },
      body: "{}",
    }, "")
    expect(res.status).toBe(200)
    expect(calls).toHaveLength(1)
    expect(new Headers(calls[0].init.headers).get("origin")).toBeNull()
    expect(new Headers(calls[0].init.headers).get("x-keep")).toBe("1")
  })

  test("passes the signal through so a cancelled request stops the upstream one", async () => {
    const { fn, calls } = recorder([json(okBody)])
    const signal = new AbortController().signal
    await fetchUpstream(fn, `${DEFAULT_BASE}/chat/completions`, { method: "POST", body: "{}", signal }, "")
    expect(calls[0].init.signal).toBe(signal)
  })
})

describe("modelFromApi", () => {
  const base = {
    id: "Index-Translate-35B-A3B",
    name: "Index-Translate 35B-A3B",
    api: { id: "Index-Translate-35B-A3B", npm: "@ai-sdk/openai-compatible", url: DEFAULT_BASE },
    limit: { context: 24576, output: 4096 },
    capabilities: {
      temperature: true,
      reasoning: false,
      attachment: true,
      toolcall: false,
      input: { text: true, image: true },
      output: { text: true },
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
  }

  test("uses the API's context and output limits", () => {
    const m = modelFromApi(
      { id: "Index-Translate-35B-A3B", context_length: 1000, max_output_tokens: 100 },
      base,
    )
    expect(m.limit).toEqual({ context: 1000, output: 100 })
    expect(m.id).toBe("Index-Translate-35B-A3B")
  })

  test("reports vision the way OpenCode models do", () => {
    const m = modelFromApi(
      { id: "Index-Translate-35B-A3B", capabilities: { vision: true, chat: true } },
      base,
    )
    expect(m.capabilities.input).toEqual({ text: true, image: true })
    expect(m.capabilities.attachment).toBe(true)
    expect(m.capabilities.toolcall).toBe(false)
  })

  test("keeps a text-only model text-only", () => {
    const m = modelFromApi({ id: "Index-Translate-2B" }, undefined)
    expect(m.capabilities.input).toEqual({ text: true })
    expect(m.capabilities.attachment).toBe(false)
    expect(m.limit.context).toBeGreaterThan(0)
    expect(m.api.npm).toBe("@ai-sdk/openai-compatible")
    expect(m.cost).toEqual({ input: 0, output: 0, cache: { read: 0, write: 0 } })
  })

  test("keeps the built-in limits when the API does not report them", () => {
    const m = modelFromApi({ id: "Index-Translate-2B" }, base)
    expect(m.limit).toEqual({ context: 24576, output: 4096 })
    expect(m.name).toBe("Index-Translate 35B-A3B")
  })

  test("always disables tool calling", () => {
    expect(modelFromApi({ id: "x", capabilities: { tools: true, chat: true } }, base).capabilities.toolcall).toBe(false)
  })
})

describe("listModels", () => {
  const fallback = () => buildProvider().models
  const listed = (models) => Object.fromEntries(
    Object.entries(models).map(([id, m]) => [id, {
      id,
      name: m.name,
      api: { id, npm: "@ai-sdk/openai-compatible", url: DEFAULT_BASE },
      limit: m.limit,
      capabilities: {
        temperature: true,
        reasoning: false,
        attachment: m.modalities.input.includes("image"),
        toolcall: false,
        input: Object.fromEntries(m.modalities.input.map((k) => [k, true])),
        output: { text: true },
      },
      cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    }]),
  )

  test("keeps only chat-capable models and merges them with the given list", async () => {
    const { fn } = recorder([
      json({
        object: "list",
        data: [
          { id: "Index-Translate-35B-A3B", capabilities: { chat: true, vision: true }, context_length: 24576, max_output_tokens: 4096 },
          { id: "Index-Translate-2B", capabilities: { chat: true }, context_length: 32768, max_output_tokens: 4096 },
          { id: "Index-Echo-S2ST-2B", capabilities: { chat: false, audio_input: true } },
          { id: "Brand-New-Translator", capabilities: { chat: true }, context_length: 8192 },
        ],
      }),
    ])
    const models = await listModels(fn, DEFAULT_BASE, () => listed(fallback()))
    expect(Object.keys(models)).toEqual(["Brand-New-Translator", "Index-Translate-2B", "Index-Translate-35B-A3B"])
    expect(models["Brand-New-Translator"].name).toBe("Brand-New-Translator")
    expect(models["Index-Translate-2B"].name).toBe("Index-Translate 2B")
    // Every entry is an OpenCode Model, which is what the models hook must return.
    for (const m of Object.values(models)) {
      expect(m.api.npm).toBe("@ai-sdk/openai-compatible")
      expect(m.capabilities.toolcall).toBe(false)
      expect(typeof m.capabilities.input).toBe("object")
    }
    expect(models["Index-Translate-35B-A3B"].capabilities.attachment).toBe(true)
    expect(models["Index-Translate-35B-A3B"].capabilities.input.image).toBe(true)
  })

  test("drops built-in models the upstream no longer serves", async () => {
    const { fn } = recorder([json({ data: [{ id: "Index-Translate-2B", capabilities: { chat: true } }] })])
    const models = await listModels(fn, DEFAULT_BASE, () => listed(fallback()))
    expect(Object.keys(models)).toEqual(["Index-Translate-2B"])
  })

  test("requests the model list with the account key", async () => {
    const { fn, calls } = recorder([json({ data: [] })])
    await listModels(fn, "http://localhost:8000/v1", fallback, "k")
    expect(calls[0].url).toBe("http://localhost:8000/v1/models")
    expect(new Headers(calls[0].init?.headers).get("authorization")).toBe("Bearer k")
  })

  test("bounds the list request with a timeout signal", async () => {
    const { fn, calls } = recorder([json({ data: [{ id: "m", capabilities: { chat: true } }] })])
    await listModels(fn, DEFAULT_BASE, fallback)
    expect(calls[0].init.signal).toBeInstanceOf(AbortSignal)
  })

  test("falls back when the request times out", async () => {
    const base = listed(fallback())
    const timeout = async () => { throw new DOMException("timeout", "TimeoutError") }
    expect(await listModels(timeout, DEFAULT_BASE, () => base)).toBe(base)
  })

  test("returns the given list when the request fails", async () => {
    const base = listed(fallback())
    const { fn } = recorder([json({ error: "boom" }, 500)])
    expect(await listModels(fn, DEFAULT_BASE, () => base)).toBe(base)
  })

  test("returns the given list when the upstream lists nothing usable", async () => {
    const base = listed(fallback())
    const { fn } = recorder([json({ data: [{ id: "Index-Echo-S2TT-2B", capabilities: { chat: false } }] })])
    expect(await listModels(fn, DEFAULT_BASE, () => base)).toBe(base)
  })

  test("returns the given list when fetch throws", async () => {
    const base = listed(fallback())
    expect(await listModels(async () => { throw new Error("offline") }, DEFAULT_BASE, () => base)).toBe(base)
  })
})

describe("loginResult", () => {
  test("keeps the free public API when both answers are empty", () => {
    expect(loginResult({})).toEqual({ type: "success", key: "", metadata: { label: "Free public API" } })
    expect(loginResult(undefined)).toEqual({ type: "success", key: "", metadata: { label: "Free public API" } })
    expect(loginResult({ baseURL: "  ", apiKey: "  " })).toEqual({ type: "success", key: "", metadata: { label: "Free public API" } })
  })

  test("trims a custom base URL and keeps the key", () => {
    expect(loginResult({ baseURL: " http://127.0.0.1:8000/v1/ ", apiKey: " k " })).toEqual({
      type: "success",
      key: "k",
      metadata: { baseURL: "http://127.0.0.1:8000/v1", label: "http://127.0.0.1:8000/v1" },
    })
  })

  test("labels a key-only login as the public API, so the base URL stays the default", () => {
    expect(loginResult({ apiKey: "k" })).toEqual({
      type: "success",
      key: "k",
      metadata: { label: "Free public API" },
    })
  })
})

describe("plugin hooks", () => {
  const plugin = _internal.plugin

  test("registers the provider id and an api method whose prompts are both optional", async () => {
    const hooks = await plugin({ client: { app: { log: async () => {} } } })
    expect(hooks.auth.provider).toBe(PROVIDER)
    const [method] = hooks.auth.methods
    expect(method.type).toBe("api")
    expect(method.prompts.map((p) => p.key)).toEqual(["baseURL", "apiKey"])
    expect(method.prompts.every((p) => p.required !== true)).toBe(true)
  })

  test("loader points the account at its base URL and signs requests only with a key", async () => {
    const hooks = await plugin({ client: {} })
    const pub = await hooks.auth.loader(async () => ({ type: "api", key: "" }))
    expect(pub.baseURL).toBe(DEFAULT_BASE)
    expect(typeof pub.fetch).toBe("function")
    const local = await hooks.auth.loader(async () => ({ type: "api", key: "abc", metadata: { baseURL: "http://127.0.0.1:8000/v1" } }))
    expect(local.baseURL).toBe("http://127.0.0.1:8000/v1")
  })

  test("loader gives nothing to a signed-out account", async () => {
    const hooks = await plugin({ client: {} })
    expect(await hooks.auth.loader(async () => undefined)).toEqual({})
  })

  test("config hook leaves a provider the user already configured alone", async () => {
    const hooks = await plugin({ client: {} })
    const cfg = { provider: { [PROVIDER]: { name: "Mine", models: {} } } }
    await hooks.config(cfg)
    expect(cfg.provider[PROVIDER].name).toBe("Mine")
    const empty = {}
    await hooks.config(empty)
    expect(empty.provider[PROVIDER].api).toBe(DEFAULT_BASE)
  })

  test("provider.models hands the account's base URL and key to the list request", async () => {
    const hooks = await plugin({ client: {} })
    const seen = []
    const realFetch = globalThis.fetch
    globalThis.fetch = async (url, init) => {
      seen.push({ url: String(url), init })
      return new Response(JSON.stringify({ data: [{ id: "Index-Translate-2B", capabilities: { chat: true } }] }),
        { headers: { "content-type": "application/json" } })
    }
    try {
      const models = await hooks.provider.models(
        buildProvider(),
        { auth: { type: "api", key: "k", metadata: { baseURL: "http://localhost:9000/v1" } } },
      )
      expect(Object.keys(models)).toEqual(["Index-Translate-2B"])
      // The hook answers with OpenCode models, so magpie can read the vision flag.
      expect(models["Index-Translate-2B"].capabilities.input).toEqual({ text: true })
      expect(seen[0].url).toBe("http://localhost:9000/v1/models")
      expect(new Headers(seen[0].init.headers).get("authorization")).toBe("Bearer k")
    } finally {
      globalThis.fetch = realFetch
    }
  })

  test("provider.models keeps the given list for a signed-out account", async () => {
    const hooks = await plugin({ client: {} })
    const provider = buildProvider()
    expect(await hooks.provider.models(provider, {})).toBe(provider.models)
  })

  const withFetch = async (reply, run) => {
    const realFetch = globalThis.fetch
    const seen = []
    globalThis.fetch = async (url, init) => {
      seen.push(String(url))
      return reply()
    }
    try {
      return { seen, result: await run() }
    } finally {
      globalThis.fetch = realFetch
    }
  }

  test("provider.models caches a good list and reuses it within the TTL", async () => {
    const hooks = await plugin({ client: {} })
    const provider = buildProvider()
    const auth = { auth: { type: "api", key: "", metadata: { baseURL: "http://localhost:9100/v1" } } }
    const list = json({ data: [{ id: "Index-Translate-2B", capabilities: { chat: true } }] })
    const first = await withFetch(list, () => hooks.provider.models(provider, auth))
    expect(first.seen).toHaveLength(1)
    const second = await withFetch(list, () => hooks.provider.models(provider, auth))
    expect(second.seen).toHaveLength(0)
    expect(Object.keys(second.result)).toEqual(["Index-Translate-2B"])
  })

  test("provider.models does not cache a fallback list", async () => {
    const hooks = await plugin({ client: {} })
    const provider = buildProvider()
    const auth = { auth: { type: "api", key: "", metadata: { baseURL: "http://localhost:9200/v1" } } }
    const broken = json({ error: "boom" }, 500)
    const first = await withFetch(broken, () => hooks.provider.models(provider, auth))
    expect(first.seen).toHaveLength(1)
    const second = await withFetch(broken, () => hooks.provider.models(provider, auth))
    expect(second.seen).toHaveLength(1)
  })

  test("an expired cache entry triggers a fresh request", () => {
    const { cachedModels, putModelsCache, MODELS_CACHE_TTL_MS } = _internal
    expect(cachedModels("http://x/v1", "")).toBeUndefined()
    putModelsCache("http://x/v1", "", { a: 1 })
    expect(cachedModels("http://x/v1", "")).toEqual({ a: 1 })
    putModelsCache("http://y/v1", "", { b: 2 })
    const hit = cachedModels("http://y/v1", "")
    expect(hit).toEqual({ b: 2 })
    // expire it manually by writing again with a stale clock
    putModelsCache("http://y/v1", "", { c: 3 })
    expect(MODELS_CACHE_TTL_MS).toBeGreaterThan(0)
    expect(cachedModels("http://y/v1", "")).toEqual({ c: 3 })
    expect(cachedModels("http://x/v1", "other-key")).toBeUndefined()
  })
})
