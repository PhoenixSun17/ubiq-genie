export type Language = 'zh' | 'en';

export interface VoiceProfile {
    refAudioPath: string;
    promptText: string;
    promptLanguage: Language;
}

/** Public catalog metadata; reference paths and transcripts stay on the server. */
export interface VoiceCatalogEntry {
    id: string;
    name: string;
    language: Language;
}

export interface TranslationOptions {
    baseUrl: string;
    model: string;
    timeoutMs: number;
}

export interface SynthesisOptions {
    baseUrl: string;
    timeoutMs: number;
}

export interface AudioOptions {
    ffmpegPath: string;
    sampleRate: number;
    timeoutMs: number;
}

export type LogFunction = (message: string) => void;
