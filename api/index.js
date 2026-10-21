import { waitUntil } from "@vercel/functions";

const DEFAULT_MODEL = "gemini-3.5-flash-lite";
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_CHARS = 1200;
const MAX_ALICE_TEXT = 1024;
const TOTAL_RESPONSE_BUDGET_MS = 4100;
const SERPER_TIMEOUT_MS = 1400;
const GEMINI_TIMEOUT_MS = 3500;
const FIRST_ATTEMPT_MS = 2200;
const FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"];

const SYSTEM_PROMPT = [
  "Ты — полезный голосовой помощник в навыке Яндекс Алисы.",
  "Отвечай естественно, по-русски, кратко и так, чтобы ответ было удобно слушать.",
  "Обычно используй 1–3 коротких предложения. Не используй Markdown, таблицы, списки с символами или длинные вступления.",
  "Если вопрос требует актуальных данных и переданы результаты поиска, опирайся на них; не выдумывай отсутствующие факты.",
  "Никогда не говори, что не можешь искать в интернете или открывать браузер. Если данных поиска нет, коротко скажи, что не удалось получить свежие данные, и предложи повторить вопрос.",
  "Если не знаешь ответа, честно скажи об этом."
].join(" ");

function aliceResponse(text, endSession = false, history = [], pending = false) {
  return {
    version: "1.0",
    response: {
      text: fitAliceText(text),
      end_session: endSession
    },
    session_state: { history, pending }
  };
}

function fitAliceText(value) {
  const text = String(value ?? "")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (text.length <= MAX_ALICE_TEXT) return text || "Не удалось подготовить ответ.";

  const candidate = text.slice(0, MAX_ALICE_TEXT - 1);
  const sentenceEnd = Math.max(
    candidate.lastIndexOf(". "),
    candidate.lastIndexOf("! "),
    candidate.lastIndexOf("? ")
  );
  if (sentenceEnd >= 300) return candidate.slice(0, sentenceEnd + 1).trim();

  const lastSpace = candidate.lastIndexOf(" ");
  return (lastSpace > 200 ? candidate.slice(0, lastSpace) : candidate).trim() + "…";
}

function isSearchIntent(text) {
  return /\b(latest|today|current|news|weather|price|exchange rate|recent|search|look up)\b|сегодня|сейчас|последн|новост|погод|курс валют|найди|поищи|актуальн|сколько стоит|свежие данные|завтра|послезавтра|прогноз|температур|дожд|биткоин|курс доллара|курс евро|результат матча|кто выиграл/i.test(text);
}

function safeHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter(item => item && (item.role === "user" || item.role === "model") && typeof item.text === "string")
    .slice(-MAX_HISTORY_MESSAGES)
    .map(item => ({ role: item.role, text: item.text.slice(0, MAX_HISTORY_MESSAGE_CHARS) }));
}

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(100, timeoutMs));
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  } finally {
    clearTimeout(timer);
  }
}

async function searchWeb(query, timeoutMs) {
  const key = process.env.SERPER_API_KEY;
  if (!key || timeoutMs < 150) return "";

  try {
    const { response, data } = await fetchJson(
      "https://google.serper.dev/search",
      {
        method: "POST",
        headers: {
          "X-API-KEY": key,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ q: query, gl: "ru", hl: "ru", num: 3 })
      },
      timeoutMs
    );

    if (!response.ok) {
      console.warn("Serper request failed with HTTP", response.status);
      return "";
    }

    const items = (Array.isArray(data.organic) ? data.organic : [])
      .slice(0, 3)
      .map((item, i) => {
        const title = String(item.title || "").slice(0, 180);
        const snippet = String(item.snippet || "").slice(0, 350);
        const link = String(item.link || "").slice(0, 240);
        return `${i + 1}. ${title}: ${snippet} ${link}`.trim();
      });

    const box = data.answerBox;
    const answerBox = box?.answer || box?.snippet || (box ? JSON.stringify(box) : "");
    if (answerBox) items.unshift(`Краткий ответ поисковика: ${String(answerBox).slice(0, 350)}`);
    return items.length ? items.join("\n") : "";
  } catch (error) {
    console.warn("Serper request did not finish in time; continuing without search.");
    return "";
  }
}

function thinkingConfigFor(model) {
  // gemini-3.8-flash thinks at "medium" by default and has no "minimal" level; keep it fast for voice.
  return model === "gemini-3.8-flash" ? { thinkingConfig: { thinkingLevel: "low" } } : {};
}

async function askGeminiModel(model, text, history, searchContext, timeoutMs) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error("MISSING_GEMINI_KEY");
  if (timeoutMs < 250) throw new Error("TIME_BUDGET_EXHAUSTED");

  const contents = safeHistory(history).map(item => ({
    role: item.role,
    parts: [{ text: item.text }]
  }));

  const prompt = searchContext
    ? `${text}\n\nРезультаты поиска в интернете:\n${searchContext}\nИспользуй их только там, где они относятся к вопросу.`
    : text;
  contents.push({ role: "user", parts: [{ text: prompt }] });

  const { response, data } = await fetchJson(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": key
      },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents,
        generationConfig: {
          temperature: 0.5,
          maxOutputTokens: 400,
          ...thinkingConfigFor(model)
        }
      })
    },
    timeoutMs
  );

  if (!response.ok) {
    const status = Number(response.status);
    console.error("Gemini request failed with HTTP", status);
    if (status === 401 || status === 403) throw new Error("GEMINI_KEY_REJECTED");
    if (status === 429) throw new Error("GEMINI_QUOTA");
    throw new Error("GEMINI_REQUEST_FAILED");
  }

  const candidate = data?.candidates?.[0];
  const answer = (candidate?.content?.parts || [])
    .map(part => typeof part.text === "string" ? part.text : "")
    .join("")
    .trim();

  if (!answer) {
    if (data?.promptFeedback?.blockReason) throw new Error("GEMINI_BLOCKED");
    throw new Error("GEMINI_EMPTY_RESPONSE");
  }
  if (candidate?.finishReason === "MAX_TOKENS") {
    console.warn("Gemini reached the output token limit.");
  }
  return answer;
}

function modelChain() {
  const configured = String(process.env.GEMINI_MODEL || DEFAULT_MODEL).trim();
  return [...new Set([configured, ...FALLBACK_MODELS])].filter(name => /^[A-Za-z0-9._-]+$/.test(name));
}

async function askGemini(text, history, searchContext, timeoutMs, firstAttemptMs = FIRST_ATTEMPT_MS) {
  const deadline = Date.now() + timeoutMs;
  const models = modelChain();
  let lastError = new Error("TIME_BUDGET_EXHAUSTED");

  for (let i = 0; i < models.length; i++) {
    const left = deadline - Date.now();
    if (left < 250) break;
    const isLast = i === models.length - 1;
    try {
      return await askGeminiModel(models[i], text, history, searchContext, isLast ? left : Math.min(firstAttemptMs, left));
    } catch (error) {
      const message = error?.message;
      if (message === "MISSING_GEMINI_KEY" || message === "GEMINI_KEY_REJECTED" || message === "GEMINI_BLOCKED") throw error;
      console.warn("Gemini model " + models[i] + " failed: " + (message || error?.name) + "; trying next model");
      lastError = error?.name === "AbortError" ? new Error("TIME_BUDGET_EXHAUSTED") : error;
    }
  }
  throw lastError;
}

function errorForUser(error) {
  switch (error?.message) {
    case "MISSING_GEMINI_KEY":
      return "Сервер пока не настроен. Проверьте переменную GEMINI_API_KEY в настройках Vercel.";
    case "GEMINI_KEY_REJECTED":
      return "Не удалось подключиться к Gemini. Проверьте API-ключ в настройках Vercel.";
    case "GEMINI_QUOTA":
      return "Сейчас превышен лимит запросов к Gemini. Попробуйте немного позже.";
    case "GEMINI_BLOCKED":
      return "Я не смогла обработать этот запрос. Попробуйте спросить иначе.";
    case "TIME_BUDGET_EXHAUSTED":
    case "AbortError":
      return "Не успела подготовить ответ. Попробуйте задать вопрос ещё раз, чуть короче.";
    default:
      return "Не получилось получить ответ от нейросети. Попробуйте ещё раз чуть позже.";
  }
}

// Answers are prepared in the background so a slow search/Gemini call does not hit Alice's 4.5 s limit.
// Results are kept in memory per session (best effort: a cold start loses them).
const jobs = new Map();
const JOB_TTL_MS = 5 * 60 * 1000;
const JOB_SEARCH_MS = 2500;
const JOB_GEMINI_MS = 7500;
const JOB_FIRST_ATTEMPT_MS = 3000;
const READY_RE = /готов|дальше|что там|ответ|давай|слуша|продолж|нашл|узнал/i;
const PENDING_TEXT = "Секунду, ещё ищу информацию. Скажи «готово», когда захочешь услышать ответ.";

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function pruneJobs() {
  const now = Date.now();
  for (const [key, job] of jobs) {
    if (now - job.at > JOB_TTL_MS) jobs.delete(key);
  }
}

async function runJob(text, history) {
  try {
    let searchContext = "";
    if (isSearchIntent(text) && process.env.SERPER_API_KEY) {
      searchContext = await searchWeb(text, JOB_SEARCH_MS);
    }
    const answer = await askGemini(text, history, searchContext, JOB_GEMINI_MS, JOB_FIRST_ATTEMPT_MS);
    return { answer };
  } catch (error) {
    console.warn("Alice job failed:", error?.message || "unknown error");
    return { error };
  }
}

async function replyFromJob(res, sessionId, job, oldHistory, waitMs) {
  const result = await Promise.race([job.promise, sleep(Math.max(0, waitMs)).then(() => null)]);

  if (!result) {
    return res.status(200).json(aliceResponse(PENDING_TEXT, false, oldHistory, true));
  }

  jobs.delete(sessionId);

  if (result.error) {
    console.warn("Alice webhook returned a fallback response:", result.error?.message || "unknown error");
    return res.status(200).json(aliceResponse(errorForUser(result.error), false, oldHistory));
  }

  const history = [
    ...oldHistory,
    { role: "user", text: job.question.slice(0, MAX_HISTORY_MESSAGE_CHARS) },
    { role: "model", text: result.answer.slice(0, MAX_HISTORY_MESSAGE_CHARS) }
  ].slice(-MAX_HISTORY_MESSAGES);

  return res.status(200).json(aliceResponse(result.answer, false, history));
}

export default async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({
      ok: true,
      service: "alice-gemini-serper",
      message: "Webhook is ready. Yandex Dialogs should send POST requests to /api."
    });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const startedAt = Date.now();
  const remainingMs = () => TOTAL_RESPONSE_BUDGET_MS - (Date.now() - startedAt);
  const body = req.body && typeof req.body === "object" ? req.body : {};
  const text = String(body?.request?.command || body?.request?.original_utterance || "").trim();
  const oldHistory = safeHistory(body?.state?.session?.history);
  const sessionId = String(body?.session?.session_id || "anon");
  const pending = body?.state?.session?.pending === true;

  if (!text) {
    return res.status(200).json(aliceResponse(
      "Привет! Я могу ответить на вопрос или найти актуальную информацию. Что хочешь узнать?",
      false,
      oldHistory
    ));
  }

  pruneJobs();

  // The user asks for the answer that is still being prepared.
  if (pending && text.length < 40 && READY_RE.test(text)) {
    const job = jobs.get(sessionId);
    if (!job) {
      return res.status(200).json(aliceResponse(
        "Не получилось сохранить ответ. Задай вопрос ещё раз.",
        false,
        oldHistory
      ));
    }
    return replyFromJob(res, sessionId, job, oldHistory, remainingMs() - 500);
  }

  // A new question: prepare the answer in the background and reply right away if it is quick enough.
  const promise = runJob(text, oldHistory);
  const job = { promise, question: text, at: Date.now() };
  jobs.set(sessionId, job);
  try {
    waitUntil(promise);
  } catch (error) {
    console.warn("waitUntil is not available:", error?.message || "unknown error");
  }
  return replyFromJob(res, sessionId, job, oldHistory, remainingMs() - 600);
}

export { aliceResponse, fitAliceText, isSearchIntent, safeHistory };
