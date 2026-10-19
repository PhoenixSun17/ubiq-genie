import fs from 'node:fs';
import path from 'node:path';

export interface SpeechTranslationConfig {
    runtimePath: string;
    profilesPath: string;
    outputSampleRate: number;
    maxQueueLength: number;
    maxPeers: number;
    requestTimeoutMs: number;
    workerStartupTimeoutMs: number;
    asr: { pythonPath: string; modelPath: string; device: 'cuda'; computeType: 'float16' };
    vad: { silenceMs: number; preRollMs: number; minSpeechMs: number; maxSpeechMs: number; threshold: number };
    ollama: { baseUrl: string; model: string };
    tts: { baseUrl: string };
    ffmpegPath: string;
}

function object(value: unknown, label: string): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
    return value as Record<string, unknown>;
}

function string(value: unknown, label: string): string {
    if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a nonempty string`);
    return value.trim();
}

function integer(value: unknown, label: string, min: number, max: number): number {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
        throw new Error(`${label} must be an integer in ${min}..${max}`);
    }
    return value;
}

function localUrl(value: unknown, label: string): string {
    const url = new URL(string(value, label));
    if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ||
        url.username || url.password || url.search || url.hash || url.pathname !== '/') {
        throw new Error(`${label} must be a loopback HTTP server origin, without credentials or a path`);
    }
    return url.origin;
}

/** Resolve paths relative to the app, and runtime-owned paths relative to runtimePath. */
export function readTranslationConfig(configPath: string): SpeechTranslationConfig {
    const raw = object(JSON.parse(fs.readFileSync(configPath, 'utf8').replace(/^\uFEFF/, '')), 'config');
    const t = object(raw.translation, 'translation');
    const asr = object(t.asr, 'asr');
    const vad = object(t.vad, 'vad');
    const ollama = object(t.ollama, 'ollama');
    const tts = object(t.tts, 'tts');
    const appPath = path.dirname(configPath);
    const runtimePath = path.resolve(appPath, string(t.runtimePath, 'runtimePath'));
    if (asr.device !== 'cuda' || asr.computeType !== 'float16') {
        throw new Error('This deployment requires ASR device=cuda and computeType=float16; no silent CPU fallback.');
    }
    const threshold = vad.threshold;
    if (typeof threshold !== 'number' || !Number.isFinite(threshold) || threshold <= 0 || threshold >= 1) {
        throw new Error('vad.threshold must be between 0 and 1');
    }
    const result: SpeechTranslationConfig = {
        runtimePath,
        profilesPath: path.resolve(appPath, string(t.profilesPath, 'profilesPath')),
        outputSampleRate: integer(t.outputSampleRate, 'outputSampleRate', 8000, 96000),
        maxQueueLength: integer(t.maxQueueLength, 'maxQueueLength', 1, 31),
        maxPeers: integer(t.maxPeers, 'maxPeers', 1, 32),
        requestTimeoutMs: integer(t.requestTimeoutMs, 'requestTimeoutMs', 1000, 600000),
        workerStartupTimeoutMs: integer(t.workerStartupTimeoutMs, 'workerStartupTimeoutMs', 1000, 1800000),
        asr: {
            pythonPath: path.resolve(runtimePath, string(asr.pythonPath, 'asr.pythonPath')),
            modelPath: path.resolve(runtimePath, string(asr.modelPath, 'asr.modelPath')),
            device: 'cuda', computeType: 'float16',
        },
        vad: {
            silenceMs: integer(vad.silenceMs, 'vad.silenceMs', 100, 5000),
            preRollMs: integer(vad.preRollMs, 'vad.preRollMs', 0, 2000),
            minSpeechMs: integer(vad.minSpeechMs, 'vad.minSpeechMs', 32, 5000),
            maxSpeechMs: integer(vad.maxSpeechMs, 'vad.maxSpeechMs', 1000, 15000),
            threshold,
        },
        ollama: { baseUrl: localUrl(ollama.baseUrl, 'ollama.baseUrl'), model: string(ollama.model, 'ollama.model') },
        tts: { baseUrl: localUrl(tts.baseUrl, 'tts.baseUrl') },
        ffmpegPath: path.resolve(runtimePath, string(t.ffmpegPath, 'ffmpegPath')),
    };
    if (result.vad.preRollMs + result.vad.minSpeechMs >= result.vad.maxSpeechMs || result.vad.silenceMs >= result.vad.maxSpeechMs) {
        throw new Error('vad.maxSpeechMs must exceed preRollMs + minSpeechMs, and silenceMs');
    }
    return result;
}
