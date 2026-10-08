/**
 * Gemini, given a YouTube URL, watches the video itself: Google fetches it, so this
 * works from cloud IPs that YouTube refuses to serve captions to.
 *
 * Env: GEMINI_API_KEY (required), GEMINI_MODEL and GEMINI_FALLBACK_MODEL (optional).
 */

const API = 'https://generativelanguage.googleapis.com/v1beta/models';
// Of the models tried, the one whose timestamps track the video; 3.8-flash dropped
// them, the lite models drifted.
const DEFAULT_MODEL = 'gemini-3.5-flash';
// Tried when the model keeps refusing a clip as RECITATION; the lite model refuses
// less, and its timestamps hold up over a clip a few minutes long.
export const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.5-flash-lite';

export interface Clip {
  startSeconds?: number;
  endSeconds?: number;
}

export interface GeminiResult {
  text: string;
  model: string;
  finishReason?: string;
}

export function hasGeminiApiKey(): boolean {
  return Boolean(process.env.GEMINI_API_KEY);
}

/** The 11-character video ID of an ID or a youtube.com / youtu.be URL. */
export function videoIdOf(videoIdOrUrl: string): string {
  const s = videoIdOrUrl.trim();
  if (/^[\w-]{11}$/.test(s)) return s;
  const m = s.match(/(?:v=|youtu\.be\/|\/shorts\/|\/embed\/|\/live\/)([\w-]{11})/);
  if (!m) throw new Error(`Not a YouTube video ID or URL: ${s}`);
  return m[1];
}

/**
 * Ask Gemini `prompt` about the video, optionally only a clip of it, or about no
 * video at all if `videoIdOrUrl` is null. A response
 * Gemini cut short (finishReason other than STOP, e.g. RECITATION) is returned, not
 * thrown, so callers can retry.
 */
export async function askGemini(
  videoIdOrUrl: string | null,
  prompt: string,
  clip: Clip = {},
  { thinkingLevel = 'minimal', model = process.env.GEMINI_MODEL || DEFAULT_MODEL } = {},
): Promise<GeminiResult> {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error('GEMINI_API_KEY is not set');

  const videoMetadata: Record<string, string> = {};
  if (clip.startSeconds != null) videoMetadata.startOffset = `${clip.startSeconds}s`;
  if (clip.endSeconds != null) videoMetadata.endOffset = `${clip.endSeconds}s`;

  const res = await fetch(`${API}/${model}:generateContent`, {
    method: 'POST',
    // The key goes in a header, never the URL, so it can't leak into error messages.
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
    body: JSON.stringify({
      contents: [{
        role: 'user',
        parts: [
          ...(videoIdOrUrl ? [{
            fileData: { fileUri: `https://www.youtube.com/watch?v=${videoIdOf(videoIdOrUrl)}` },
            ...(Object.keys(videoMetadata).length && { videoMetadata }),
          }] : []),
          { text: prompt },
        ],
      }],
      generationConfig: {
        // Not 0: at 0, transcripts of well-known videos are refused as RECITATION more often.
        temperature: 1,
        // Low resolution is ~3x cheaper per second of video; the audio is unaffected.
        ...(videoIdOrUrl && { mediaResolution: 'MEDIA_RESOLUTION_LOW' }),
        thinkingConfig: { thinkingLevel },
      },
    }),
  });

  const data: any = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`Gemini ${res.status}: ${data?.error?.message || res.statusText}`);
  }
  const candidate = data.candidates?.[0];
  const text = (candidate?.content?.parts ?? []).map((p: any) => p.text ?? '').join('');
  const finishReason = candidate?.finishReason || data.promptFeedback?.blockReason;
  console.error(`[gemini] model=${model} clip=${JSON.stringify(clip)} finish=${finishReason} usage=${JSON.stringify(data.usageMetadata)}`);
  return { text, model, finishReason };
}
