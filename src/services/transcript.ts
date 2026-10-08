import { YoutubeTranscript } from "youtube-transcript";
import { TranscriptParams, SearchTranscriptParams, AskVideoParams, SummarizeParams } from '../types.js';
import { askGemini, hasGeminiApiKey, videoIdOf, FALLBACK_MODEL } from './gemini.js';
import { hasConfiguredYouTubeApiKey, withYouTubeClient } from './youtube-client.js';

// "Clean" rather than verbatim, and temperature 1 in askGemini: Gemini refuses
// verbatim transcripts of well-known videos as RECITATION far more often.
const TRANSCRIBE_PROMPT = `Write a clean transcript of the speech in this video, in the language it is spoken in: keep the speaker's words, but drop filler words and false starts.
Start a new paragraph at each change of speaker or topic, prefixed with its [mm:ss] (or [h:mm:ss]) timestamp, counted from the start of the whole video.
Label speakers by name when the video makes them clear, otherwise as Speaker 1, Speaker 2, ...
Mark significant non-speech audio briefly in brackets, e.g. [music]. Output only the transcript.`;

const SUMMARY_FORMAT = `First a 2-3 sentence overview, then the main points as a list, each with the [mm:ss] (or [h:mm:ss]) timestamp, counted from the start of the whole video, where it is discussed.`;

// Summaries: up to this long, one call on the whole span. Gemini skims longer
// videos (a one-hour summary covered only its first 15 minutes), so they are
// summarized in clips in parallel, and the clips' notes merged.
const SUMMARY_SINGLE_SECONDS = 1200;
const SUMMARY_CLIP_SECONDS = 600;
const SUMMARY_MAX_SPAN_SECONDS = 2 * 3600;  // ~30s per call

// Clips are transcribed in parallel, so a call takes about as long as one clip
// (~10s), well inside CloudFront's 30s origin timeout.
const CLIP_SECONDS = 300;
// Per call; longer spans are continued with startSeconds.
const MAX_SPAN_SECONDS = 3600;
const ATTEMPTS = 3;

const clock = (t: number) => {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  return `${h ? `${h}:` : ''}${String(m).padStart(h ? 2 : 1, '0')}:${String(s).padStart(2, '0')}`;
};

/** Gemini sometimes writes [112:13] for [1:52:13]. */
const fixTimestamps = (text: string) =>
  text.replace(/\[(\d{3,}):(\d{2})\]/g, (_, m, sec) => `[${clock(Number(m) * 60 + Number(sec))}]`);

/** The video's length in seconds from the YouTube Data API, or null if unknown. */
async function durationSeconds(videoId: string): Promise<number | null> {
  if (!hasConfiguredYouTubeApiKey()) return null;
  const res = await withYouTubeClient((youtube) => youtube.videos.list({ part: ['contentDetails'], id: [videoId] }));
  const iso = res.data.items?.[0]?.contentDetails?.duration;
  const m = iso?.match(/^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$/);
  if (!m) return null;
  const [d, h, min, sec] = m.slice(1).map((x) => Number(x || 0));
  return d * 86400 + h * 3600 + min * 60 + sec || null;  // live streams report P0D
}

/**
 * Gemini's answer for one clip, retrying its refusals (RECITATION) and failures, the
 * last time on the fallback model.
 */
async function clipWithRetries(videoId: string, prompt: string, start?: number, end?: number, thinkingLevel = 'minimal') {
  let last = '';
  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    try {
      const model = attempt === ATTEMPTS - 1 ? FALLBACK_MODEL : undefined;
      const r = await askGemini(videoId, prompt, { startSeconds: start, endSeconds: end }, { thinkingLevel, model });
      if (r.finishReason === 'STOP' && r.text.trim()) return { text: fixTimestamps(r.text.trim()), model: r.model };
      last = r.finishReason || 'empty response';
    } catch (error) {
      last = error instanceof Error ? error.message : String(error);
    }
  }
  const span = start != null ? `${clock(start)}–${end != null ? clock(end) : 'end'}` : 'video';
  return { text: `[${span}: no output, Gemini: ${last}]`, failed: `${span}: ${last}` };
}

type ClipResult = Awaited<ReturnType<typeof clipWithRetries>>;

const modelsOf = (parts: ClipResult[]) => [...new Set(parts.flatMap((p) => ('model' in p ? [p.model] : [])))];
const failuresOf = (parts: ClipResult[]) => parts.flatMap((p) => ('failed' in p ? [p.failed] : []));

/** The span [from, to) of a video a call covers, at most `max` seconds, and its length. */
async function span(id: string, startSeconds: number | undefined, endSeconds: number | undefined, max: number) {
  const duration = await durationSeconds(id).catch(() => null);
  const from = startSeconds ?? 0;
  const to = Math.min(endSeconds ?? duration ?? Infinity, from + max);
  if (to <= from) throw new Error(`startSeconds is past the end of the video (${duration}s)`);
  return {
    from, to, duration,
    ...(Number.isFinite(to) && { span: `${clock(from)}–${clock(to)}${duration ? ` of ${clock(duration)}` : ''}` }),
    ...(duration != null && to < duration && endSeconds == null && { next: `continues: call again with startSeconds=${to}` }),
  };
}

/** [start, end) pieces of at most `size` seconds; one unbounded piece if the end is unknown. */
function clips(from: number, to: number, size: number, startSeconds?: number, endSeconds?: number) {
  const out: [number | undefined, number | undefined][] = [];
  if (!Number.isFinite(to)) out.push([startSeconds, endSeconds]);
  else for (let t = from; t < to; t += size) out.push([t, Math.min(t + size, to)]);
  return out;
}

/**
 * Service for interacting with YouTube video transcripts
 */
export class TranscriptService {
  // No YouTube API key needed for transcripts, but we'll implement the same pattern
  private initialized = false;

  constructor() {
    // No initialization needed
  }

  private initialize() {
    if (this.initialized) return;
    // No API key needed for transcripts, but we'll check if language is set
    this.initialized = true;
  }

  /**
   * Get the transcript of a YouTube video: by Gemini when GEMINI_API_KEY is set
   * (works from cloud IPs, which YouTube blocks from fetching captions), otherwise
   * YouTube's own captions.
   */
  async getTranscript({ videoId, language, startSeconds, endSeconds }: TranscriptParams): Promise<any> {
    try {
      this.initialize();

      if (hasGeminiApiKey()) {
        const id = videoIdOf(videoId);
        const prompt = language
          ? `${TRANSCRIBE_PROMPT}\nIf the speech is not in the language with code "${language}", translate it into that language.`
          : TRANSCRIBE_PROMPT;
        const { from, to, span: covered, next } = await span(id, startSeconds, endSeconds, MAX_SPAN_SECONDS);
        const parts = await Promise.all(
          clips(from, to, CLIP_SECONDS, startSeconds, endSeconds).map(([a, b]) => clipWithRetries(id, prompt, a, b)));
        const models = modelsOf(parts), failed = failuresOf(parts);
        return {
          videoId: id,
          source: `gemini${models.length ? ` (${models.join(', ')})` : ''}, cleaned of filler words; timestamps approximate`,
          ...(covered && { span: covered }),
          ...(failed.length && { failedClips: failed }),
          ...(next && { next: `transcript ${next}` }),
          transcript: parts.map((p) => p.text).join('\n\n'),
        };
      }

      language = language || process.env.YOUTUBE_TRANSCRIPT_LANG || 'en';
      const transcript = await YoutubeTranscript.fetchTranscript(videoId, { lang: language });
      
      return {
        videoId,
        language,
        transcript
      };
    } catch (error) {
      throw new Error(`Failed to get transcript: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Ask Gemini a question about a video (speech and visuals).
   */
  async askVideo({ videoId, question, startSeconds, endSeconds }: AskVideoParams): Promise<any> {
    if (!hasGeminiApiKey()) throw new Error('videos_askVideo needs GEMINI_API_KEY');
    // Some thinking: answers need more reasoning than transcription.
    const result = await askGemini(videoId, question, { startSeconds, endSeconds }, { thinkingLevel: 'low' });
    if (result.finishReason !== 'STOP' || !result.text) {
      throw new Error(`Gemini gave no answer (${result.finishReason || 'empty response'})`);
    }
    return { videoId: videoIdOf(videoId), source: `gemini (${result.model})`, answer: result.text };
  }

  /**
   * Summarize a video with Gemini, straight from the video (no transcript first).
   */
  async summarize({ videoId, language, startSeconds, endSeconds }: SummarizeParams): Promise<any> {
    if (!hasGeminiApiKey()) throw new Error('videos_summarize needs GEMINI_API_KEY');
    const id = videoIdOf(videoId);
    const lang = language
      ? `Write in the language with code "${language}".`
      : 'Write in the language the video is spoken in.';
    const { from, to, span: covered, next } = await span(id, startSeconds, endSeconds, SUMMARY_MAX_SPAN_SECONDS);

    let summary: string, parts: ClipResult[];
    if (to - from <= SUMMARY_SINGLE_SECONDS || !Number.isFinite(to)) {
      parts = [await clipWithRetries(id, `Summarize this video. ${SUMMARY_FORMAT} ${lang}`, startSeconds, endSeconds, 'low')];
      summary = parts[0].text;
    } else {
      parts = await Promise.all(clips(from, to, SUMMARY_CLIP_SECONDS).map(([a, b]) => clipWithRetries(id,
        `This clip runs from ${clock(a!)} to ${clock(b!)} of the video. List its main points, each with its [mm:ss] (or [h:mm:ss]) timestamp, which must lie between ${clock(a!)} and ${clock(b!)}. Name who says what when it is clear. ${lang} Output only the list.`,
        a, b)));
      const notes = parts.map((p) => p.text).join('\n\n');
      const merged = await askGemini(null,
        `Below are notes on consecutive parts of one video, ${covered}, with timestamps. Summarize the video from them. ${SUMMARY_FORMAT} Keep the timestamps from the notes. Group related points, and keep the list to the 10-25 most important. ${lang} Output only the summary.\n\n${notes}`,
        {}, { thinkingLevel: 'low' });
      // If the merge fails, the clip notes are still a usable (longer) summary.
      summary = merged.finishReason === 'STOP' && merged.text.trim() ? fixTimestamps(merged.text.trim()) : notes;
    }
    const models = modelsOf(parts), failed = failuresOf(parts);
    return {
      videoId: id,
      source: `gemini${models.length ? ` (${models.join(', ')})` : ''}, from the video itself`,
      ...(covered && { span: covered }),
      ...(failed.length && { failedClips: failed }),
      ...(next && { next: `video ${next}` }),
      summary,
    };
  }

  /**
   * Search within a transcript
   */
  async searchTranscript({ 
    videoId, 
    query,
    language = process.env.YOUTUBE_TRANSCRIPT_LANG || 'en' 
  }: SearchTranscriptParams): Promise<any> {
    try {
      this.initialize();
      
      const transcript = await YoutubeTranscript.fetchTranscript(videoId, { lang: language });

      // Search through transcript for the query
      const matches = transcript.filter(item => 
        item.text.toLowerCase().includes(query.toLowerCase())
      );
      
      return {
        videoId,
        query,
        matches,
        totalMatches: matches.length
      };
    } catch (error) {
      throw new Error(`Failed to search transcript: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  /**
   * Get transcript with timestamps
   */
  async getTimestampedTranscript({ 
    videoId, 
    language = process.env.YOUTUBE_TRANSCRIPT_LANG || 'en' 
  }: TranscriptParams): Promise<any> {
    try {
      this.initialize();
      
      const transcript = await YoutubeTranscript.fetchTranscript(videoId, { lang: language });

      // Format timestamps in human-readable format
      const timestampedTranscript = transcript.map(item => {
        const seconds = item.offset / 1000;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        const formattedTime = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        
        return {
          timestamp: formattedTime,
          text: item.text,
          startTimeMs: item.offset,
          durationMs: item.duration
        };
      });
      
      return {
        videoId,
        language,
        timestampedTranscript
      };
    } catch (error) {
      throw new Error(`Failed to get timestamped transcript: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}