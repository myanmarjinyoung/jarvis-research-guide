import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { promises as fs } from 'fs';
import path from 'path';
import type { Brief, BriefSource } from '@/lib/brief/types';
import { checkOrigin, takeDailyBudget, takeRateLimit } from '@/lib/server/guard';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 180;

const PRIMARY_MODEL = process.env.JARVIS_BRIEF_MODEL || 'claude-sonnet-5-5';
const FALLBACK_MODEL = 'claude-haiku-4-5';
/** Fast first answer (one search, two sentences) — spoken while the card is built. */
const QUICK_MODEL = process.env.JARVIS_QUICK_MODEL || 'claude-haiku-4-5';
const CACHE_FILE = path.join(process.cwd(), 'data', 'brief-cache.json');
/** Market answers go stale — re-research after this many hours. */
const CACHE_TTL_MS = Number(process.env.CACHE_TTL_HOURS || 6) * 3600 * 1000;
/** Fallback when the disk is read-only (e.g. serverless hosting). */
const memoryCache: CacheMap = {};

function systemPrompt(today: string): string {
  return `당신은 증권사 리서치센터 행사장에서 일반 고객의 질문에 답하는 AI 안내원 "자비스"입니다.
오늘 날짜: ${today}

# 원칙
- 상대는 주식을 잘 모르는 일반 고객입니다. 호칭은 "고객님". 쉬운 말로, 전문용어는 바로 풀어서.
- 특정 종목의 매수·매도 권유, 목표가 제시, 수익 보장 표현은 절대 하지 않습니다. 원리와 확인할 점을 설명합니다.
- 오늘·최근 시장 상황이 필요한 질문은 web_search로 실제 자료를 찾아 근거로 씁니다.
- 숫자(지수, 등락률, 금리, 환율 등)는 검색 결과에서 확인된 것만 씁니다. 확인 못 한 숫자는 지어내지 말고 빼세요.
- 개념 질문(PER이 뭔가요 등)은 검색 없이 답해도 됩니다.

# 차트 (중요)
- 종목·주가·지수(코스피, 코스닥, 나스닥 등)·금리·환율·유가·원자재·업종에 관한 질문이면 chart 를 반드시 넣습니다.
  web_search 로 최근 흐름 수치(예: 최근 1~3개월 주별 종가, 최근 2주 일별 종가, 최근 분기별 실적)를 찾아 5~20개 점으로 만듭니다.
  한 번에 못 찾으면 검색어를 바꿔 다시 찾으세요 ("코스피 일별 종가", "삼성전자 주가 추이" 등).
- 개념 질문도 실제 사례 수치로 보여 줄 수 있으면 차트를 넣습니다 (예: 금리 인상기 코스피 흐름, 환율과 외국인 순매수).
- 수치는 검색 결과에서 확인한 것만. 정말 하나도 못 찾았을 때만 null.

# 출력 형식
조사가 끝나면 마지막 답변에 아래 JSON 하나만 \`\`\`json 코드블록으로 출력합니다. 다른 설명 문장은 쓰지 않습니다.
{
  "spoken": "대화하듯 들려줄 짧은 답 2문장(90자 이내). 결론 먼저, 자세한 건 화면 카드에 있으니 숫자 나열 금지. 마크다운·이모지 없이 자연스러운 존댓말.",
  "title": "카드 제목 (15자 이내)",
  "summary": "한 줄 핵심 답 (40자 이내)",
  "points": [{"head": "핵심 포인트 제목 (12자 이내)", "body": "설명 1~2문장"}],
  "flow": ["원인", "→ 다음 단계", "→ 결과"],
  "stats": [{"label": "지표 이름", "value": "값", "change": "+1.2%", "tone": "up|down|flat"}],
  "chart": {"kind": "line|bar", "title": "차트 제목", "unit": "단위", "series": [{"label": "x축 라벨", "value": 숫자}], "note": "자료 기준"} 또는 null,
  "checklist": ["고객이 스스로 확인해 볼 것 (한 줄)"]
}
- 키 순서를 위와 똑같이 지킵니다 (spoken 이 맨 앞).
- points 3~4개, flow 3~5단계(인과 흐름이 없는 질문이면 빈 배열), stats 0~4개, checklist 2~4개.
- chart 는 검색으로 확인한 실제 수치가 4개 이상 있을 때만 넣고, 아니면 null.
- flow 각 단계는 "→" 없이 짧은 명사구로 씁니다.`;
}

interface BriefRequest {
  question: string;
  fresh?: boolean;
  /** Recent conversation lines, for follow-ups like "그럼 삼성전자는?". */
  context?: string[];
}

type CacheMap = Record<string, Brief>;

function cacheKey(q: string): string {
  return q.replace(/\s+/g, ' ').replace(/[?？.!]+$/g, '').trim().toLowerCase();
}

async function readCache(): Promise<CacheMap> {
  try {
    return { ...(JSON.parse(await fs.readFile(CACHE_FILE, 'utf8')) as CacheMap), ...memoryCache };
  } catch {
    return { ...memoryCache };
  }
}

async function writeCache(map: CacheMap): Promise<void> {
  Object.assign(memoryCache, map);
  try {
    await fs.mkdir(path.dirname(CACHE_FILE), { recursive: true });
    await fs.writeFile(CACHE_FILE, JSON.stringify(map, null, 2), 'utf8');
  } catch {
    /* read-only disk — memory cache only */
  }
}

type Preview = { spoken?: string; title?: string; summary?: string };
type StreamEvent =
  | { type: 'preview'; preview: Preview }
  | { type: 'brief'; brief: Brief }
  | { type: 'error'; error: string };

/** Raw API errors are English JSON — the card shows a plain Korean line instead
 *  (the raw error still goes to the server log). */
function friendlyError(err: unknown): string {
  if (err instanceof Anthropic.APIError) {
    if (/credit balance/i.test(err.message)) return 'AI 사용 크레딧이 모두 소진되었습니다. 관리자가 충전하면 바로 다시 됩니다.';
    if (err.status === 401 || err.status === 403) return 'AI 서비스 인증에 실패했습니다. API 키 설정을 확인해 주세요.';
    if (err.status === 429) return '지금 요청이 많아 잠시 막혔습니다. 잠시 뒤 다시 시도해 주세요.';
    if (err.status === 529 || (err.status ?? 0) >= 500) return 'AI 서비스가 일시적으로 혼잡합니다. 잠시 뒤 다시 시도해 주세요.';
    return '조사 요청이 처리되지 않았습니다. 잠시 뒤 다시 시도해 주세요.';
  }
  if (err instanceof Anthropic.APIConnectionError) return '인터넷 연결이 불안정합니다. 연결을 확인해 주세요.';
  if (err instanceof Error && /[가-힣]/.test(err.message)) return err.message.slice(0, 200);
  return '조사 중 문제가 생겼습니다. 잠시 뒤 다시 시도해 주세요.';
}

function ndjson(run: (send: (e: StreamEvent) => void) => Promise<void>): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (e: StreamEvent) => controller.enqueue(enc.encode(JSON.stringify(e) + '\n'));
      try {
        await run(send);
      } catch (err) {
        console.error('[brief]', err);
        send({ type: 'error', error: friendlyError(err) });
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: { 'content-type': 'application/x-ndjson; charset=utf-8', 'cache-control': 'no-cache' },
  });
}

export async function POST(req: Request) {
  let body: BriefRequest;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'invalid json' }, { status: 400 });
  }
  const question = (body.question ?? '').toString().trim().slice(0, 300);
  if (!question) {
    return NextResponse.json({ ok: false, error: 'empty question' }, { status: 400 });
  }
  const originError = checkOrigin(req);
  if (originError) return NextResponse.json({ ok: false, error: originError }, { status: 403 });

  const context = Array.isArray(body.context)
    ? body.context.filter((l): l is string => typeof l === 'string').slice(-6).map((l) => l.slice(0, 300))
    : [];
  // Answers that depend on the conversation aren't reusable — no cache for them.
  const useCache = context.length === 0;
  const key = cacheKey(question);

  return ndjson(async (send) => {
    const saved = useCache ? (await readCache())[key] : undefined;
    if (saved && !body.fresh && Date.now() - new Date(saved.asOf).getTime() < CACHE_TTL_MS) {
      send({ type: 'brief', brief: { ...saved, cached: true } });
      return;
    }
    try {
      await research(send);
    } catch (err) {
      // API down (no credit, outage) — an older saved answer beats an error card.
      if (!saved) throw err;
      console.error('[brief] research failed, serving saved answer', err);
      send({ type: 'brief', brief: { ...saved, cached: true } });
    }
  });

  async function research(send: (e: StreamEvent) => void) {
    // Only paid research counts toward the limits; saved answers are free.
    const limited = takeRateLimit(req) ?? takeDailyBudget();
    if (limited) throw new Error(limited);
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY 미설정');
    const client = new Anthropic({ apiKey });
    const onPreview = (p: Preview) => send({ type: 'preview', preview: p });

    // Two tracks at once: a quick spoken answer (~2–4s) and the full card (~15s).
    // The client speaks whichever spoken line arrives first.
    void quickAnswer(client, question, context)
      .then((spoken) => {
        if (spoken) onPreview({ spoken });
      })
      .catch(() => {
        /* the full card's own spoken line is the fallback */
      });

    let brief: Brief;
    try {
      brief = await runBrief(client, PRIMARY_MODEL, question, context, onPreview);
    } catch (err) {
      if (err instanceof Anthropic.APIError && (err.status === 404 || /model/i.test(err.message))) {
        brief = await runBrief(client, FALLBACK_MODEL, question, context, onPreview);
      } else {
        throw err;
      }
    }
    send({ type: 'brief', brief });
    if (useCache) {
      const map = await readCache();
      map[key] = brief;
      await writeCache(map);
    }
  }
}

async function quickAnswer(client: Anthropic, question: string, context: string[]): Promise<string | null> {
  const today = new Date().toLocaleDateString('ko-KR', { timeZone: 'Asia/Seoul' });
  const content = context.length ? `앞선 대화:\n${context.join('\n')}\n\n지금 질문: ${question}` : question;
  const res = await client.messages.create({
    model: QUICK_MODEL,
    max_tokens: 300,
    system: `당신은 증권사 리서치센터 행사장의 AI 안내원 "자비스"입니다. 오늘 날짜: ${today}.
일반 고객과 대화하듯 2문장(90자 안팎) 존댓말로 바로 답합니다. 결론 먼저, 쉬운 말로.
최근·오늘 시장 상황이면 web_search 를 한 번 하고, 확인한 사실만 말합니다. 숫자를 말할 땐 기간을 분명히 ("오늘은 ~", "이번 달 들어 ~"). 확인 못 한 숫자는 말하지 않습니다.
특정 종목 매수·매도 권유, 목표가, 수익 보장 표현은 하지 않습니다.
마크다운·이모지·목록 없이 문장만 출력합니다. 자세한 자료는 화면에 띄워진다는 점을 알고 있습니다.`,
    tools: [{ type: 'web_search_20250305', name: 'web_search', max_uses: 1 } as unknown as Anthropic.Tool],
    messages: [{ role: 'user', content }],
  });
  // With a search, the answer is the text written after the last tool result.
  const blocks = res.content;
  let lastTool = -1;
  blocks.forEach((b, i) => {
    if (b.type !== 'text') lastTool = i;
  });
  const text = blocks
    .slice(lastTool + 1)
    .filter((b): b is Anthropic.TextBlock => b.type === 'text')
    .map((b) => b.text)
    .join('')
    .replace(/\s+/g, ' ')
    .trim();
  return text.length >= 8 ? text.slice(0, 260) : null;
}

/** Pull finished string fields out of a JSON object that is still being written. */
function partialField(text: string, field: string): string | undefined {
  const m = text.match(new RegExp(`"${field}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"`));
  if (!m) return undefined;
  try {
    return JSON.parse(`"${m[1]}"`) as string;
  } catch {
    return undefined;
  }
}

async function runBrief(
  client: Anthropic,
  model: string,
  question: string,
  context: string[],
  onPreview: (p: Preview) => void,
): Promise<Brief> {
  const today = new Date().toLocaleDateString('ko-KR', { timeZone: 'Asia/Seoul' });
  const content = context.length
    ? `앞선 대화:\n${context.join('\n')}\n\n지금 질문: ${question}`
    : question;
  const messages: Anthropic.MessageParam[] = [{ role: 'user', content }];
  const sources: BriefSource[] = [];
  const seen = new Set<string>();
  let finalText = '';
  const preview: Preview = {};

  for (let i = 0; i < 6; i++) {
    // Stream so the spoken answer can go out while the rest of the card is written.
    let acc = '';
    const stream = client.messages.stream({
      model,
      max_tokens: 3000,
      system: systemPrompt(today),
      tools: [
        { type: 'web_search_20250305', name: 'web_search', max_uses: 6 } as unknown as Anthropic.Tool,
      ],
      messages,
    });
    stream.on('text', (delta) => {
      acc += delta;
      let changed = false;
      for (const f of ['spoken', 'title', 'summary'] as const) {
        if (preview[f]) continue;
        const v = partialField(acc, f);
        if (v) {
          preview[f] = v;
          changed = true;
        }
      }
      if (changed && preview.spoken) onPreview({ ...preview });
    });
    const response = await stream.finalMessage();
    collectSources(response, sources, seen);
    finalText = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === 'text')
      .map((b) => b.text)
      .join('\n');
    if (response.stop_reason === 'pause_turn') {
      messages.push({ role: 'assistant', content: response.content });
      continue;
    }
    break;
  }

  const parsed = parseBriefJson(finalText);
  if (!parsed) throw new Error('조사 결과를 카드로 정리하지 못했습니다.');
  return {
    title: str(parsed.title, question).slice(0, 40),
    summary: str(parsed.summary, ''),
    spoken: str(parsed.spoken, str(parsed.summary, '조사를 마쳤습니다.')),
    points: arr(parsed.points)
      .map((p) => ({ head: str(p?.head, ''), body: str(p?.body, '') }))
      .filter((p) => p.head || p.body)
      .slice(0, 5),
    flow: arr(parsed.flow).map((s) => str(s, '').replace(/^→\s*/, '')).filter(Boolean).slice(0, 6),
    stats: arr(parsed.stats)
      .map((s) => ({
        label: str(s?.label, ''),
        value: str(s?.value, ''),
        change: s?.change ? str(s.change, '') : undefined,
        tone: s?.tone === 'up' || s?.tone === 'down' ? s.tone : 'flat',
      }))
      .filter((s) => s.label && s.value)
      .slice(0, 4),
    chart: normalizeChart(parsed.chart),
    checklist: arr(parsed.checklist).map((s) => str(s, '')).filter(Boolean).slice(0, 5),
    sources: sources.slice(0, 6),
    asOf: new Date().toISOString(),
  };
}

/* eslint-disable @typescript-eslint/no-explicit-any */
function parseBriefJson(text: string): any | null {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced?.[1] ?? text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

function normalizeChart(c: any): Brief['chart'] {
  if (!c || typeof c !== 'object') return null;
  const series = arr(c.series)
    .map((p) => ({ label: str(p?.label, ''), value: Number(p?.value) }))
    .filter((p) => p.label && Number.isFinite(p.value));
  if (series.length < 2) return null;
  return {
    kind: c.kind === 'bar' ? 'bar' : 'line',
    title: str(c.title, ''),
    unit: c.unit ? str(c.unit, '') : undefined,
    series: series.slice(0, 24),
    note: c.note ? str(c.note, '') : undefined,
  };
}

function arr(v: any): any[] {
  return Array.isArray(v) ? v : [];
}

function str(v: any, fallback: string): string {
  return typeof v === 'string' && v.trim() ? v.trim() : fallback;
}

function collectSources(response: Anthropic.Message, out: BriefSource[], seen: Set<string>) {
  for (const raw of response.content) {
    const block = raw as unknown as Record<string, any>;
    if (block.type !== 'web_search_tool_result' || !Array.isArray(block.content)) continue;
    for (const item of block.content) {
      const url = typeof item.url === 'string' ? item.url : null;
      if (!url || seen.has(url)) continue;
      seen.add(url);
      out.push({ title: typeof item.title === 'string' && item.title ? item.title : url, url });
    }
  }
}
/* eslint-enable @typescript-eslint/no-explicit-any */
