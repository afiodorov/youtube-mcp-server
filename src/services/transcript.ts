import { YoutubeTranscript } from "youtube-transcript";
import { TranscriptParams, SearchTranscriptParams, AskVideoParams } from '../types.js';
import { askGemini, hasGeminiApiKey, videoIdOf, FALLBACK_MODEL } from './gemini.js';
import { hasConfiguredYouTubeApiKey, withYouTubeClient } from './youtube-client.js';

// "Clean" rather than verbatim, and temperature 1 in askGemini: Gemini refuses
// verbatim transcripts of well-known videos as RECITATION far more often.
const TRANSCRIBE_PROMPT = `Write a clean transcript of the speech in this video, in the language it is spoken in: keep the speaker's words, but drop filler words and false starts.
Start a new paragraph at each change of speaker or topic, prefixed with its [mm:ss] (or [h:mm:ss]) timestamp, counted from the start of the whole video.
Label speakers by name when the video makes them clear, otherwise as Speaker 1, Speaker 2, ...
Mark significant non-speech audio briefly in brackets, e.g. [music]. Output only the transcript.`;

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
 * One clip's transcript, retrying Gemini's refusals (RECITATION) and failures, the
 * last time on the fallback model.
 */
async function transcribeClip(videoId: string, prompt: string, start?: number, end?: number) {
  let last = '';
  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    try {
      const model = attempt === ATTEMPTS - 1 ? { model: FALLBACK_MODEL } : {};
      const r = await askGemini(videoId, prompt, { startSeconds: start, endSeconds: end }, model);
      if (r.finishReason === 'STOP' && r.text.trim()) return { text: r.text.trim(), model: r.model };
      last = r.finishReason || 'empty response';
    } catch (error) {
      last = error instanceof Error ? error.message : String(error);
    }
  }
  const span = start != null ? `${clock(start)}–${end != null ? clock(end) : 'end'}` : 'video';
  return { text: `[${span}: not transcribed, Gemini: ${last}]`, failed: `${span}: ${last}` };
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
        const duration = await durationSeconds(id).catch(() => null);
        const from = startSeconds ?? 0;
        const to = Math.min(endSeconds ?? duration ?? Infinity, from + MAX_SPAN_SECONDS);
        if (to <= from) throw new Error(`startSeconds is past the end of the video (${duration}s)`);

        // Unknown length (no YouTube key, or a live stream): one call for the whole span.
        const clips: [number | undefined, number | undefined][] = [];
        if (!Number.isFinite(to)) clips.push([startSeconds, endSeconds]);
        else for (let t = from; t < to; t += CLIP_SECONDS) clips.push([t, Math.min(t + CLIP_SECONDS, to)]);

        const parts = await Promise.all(clips.map(([a, b]) => transcribeClip(id, prompt, a, b)));
        const failed = parts.flatMap((p) => ('failed' in p ? [p.failed] : []));
        const models = [...new Set(parts.flatMap((p) => ('model' in p ? [p.model] : [])))];
        const more = duration != null && to < duration && endSeconds == null;
        return {
          videoId: id,
          source: `gemini${models.length ? ` (${models.join(', ')})` : ''}, cleaned of filler words; timestamps approximate`,
          ...(Number.isFinite(to) && { span: `${clock(from)}–${clock(to)}${duration ? ` of ${clock(duration)}` : ''}` }),
          ...(failed.length && { failedClips: failed }),
          ...(more && { next: `transcript continues: call again with startSeconds=${to}` }),
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