import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { setTimeout as delay } from 'node:timers/promises';
import { NetworkId, NetworkScene } from 'ubiq-server/ubiq';
import { ServiceController } from '../../components/service';
import type { SpeechTranslationConfig } from './config';
import { createWhisperProvider } from './provider';
import { VoiceProfileStore } from './profiles';
import { JsonLineDecoder } from './protocol';
import { translate, synthesize } from './clients';
import { splitPcm, wavToPcm } from './audio';
import type { Language, VoiceProfile } from './types';

interface Job {
    requestId: string;
    peerUuid: string;
    endedAt: number;
    durationMs: number;
    voice: VoiceProfile;
    controller: AbortController;
}

interface Transcript {
    requestId: string;
    peerUuid: string;
    text: string;
    language: string;
    languageProbability: number;
}

interface PendingTranscript {
    job: Job;
    resolve: (value: Transcript) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
}

export class SpeechTranslationService extends ServiceController {
    private readonly profiles: VoiceProfileStore;
    private readonly decoder: JsonLineDecoder;
    private readonly peers = new Set<string>();
    private readonly warned = new Set<string>();
    private readonly resetAfterDrain = new Set<string>();
    private readonly queue: Job[] = [];
    private readonly lifetime = new AbortController();
    private readonly diagnosticsEnabled = process.env.SPEECH_TRANSLATION_DIAGNOSTICS === '1';
    private readonly inputStats = new Map<string, { frames: number; samples: number; squares: number; peak: number; lastAt: number }>();
    private diagnosticsTimer?: NodeJS.Timeout;
    private active?: Job;
    private pending?: PendingTranscript;
    private workerReady = false;
    private backpressured = false;
    private processing = false;
    private stopped = false;
    private startupTimer?: NodeJS.Timeout;
    private readyResolve?: () => void;
    private readyReject?: (error: Error) => void;

    constructor(private readonly scene: NetworkScene, private readonly options: SpeechTranslationConfig) {
        // Start the singleton explicitly after framing and lifecycle handlers are installed.
        super(scene, 'SpeechTranslation');
        this.profiles = new VoiceProfileStore(options.profilesPath, (message) => this.log(message, 'warning'));
        this.decoder = new JsonLineDecoder(
            (message) => this.onWorkerMessage(message),
            (error) => this.failWorker(error),
        );
        this.on('data', (data: Buffer) => this.decoder.push(data));
        this.on('close', (code: number | null, signal: string | null) => {
            if (!this.stopped) this.failWorker(new Error(`ASR worker exited (${code ?? signal}); restart the app after correcting the cause.`));
        });
        this.profiles.on('changed', (uuids: string[]) => {
            for (const uuid of uuids) {
                this.warned.delete(`profile:${uuid}`);
                if (!this.peers.has(uuid)) continue;
                if (!this.profiles.snapshot(uuid)) this.cancelPeerJobs(uuid);
                this.send({ type: 'reset_peer', peerUuid: uuid });
                this.log(`Voice profile ${this.profiles.snapshot(uuid) ? 'available/updated' : 'removed'} for ${uuid}`);
            }
        });
    }

    async start(): Promise<void> {
        for (const required of [this.options.asr.pythonPath, this.options.asr.modelPath, this.options.ffmpegPath]) {
            if (!fs.existsSync(required)) throw new Error(`Missing runtime file/directory: ${required}. Run scripts/setup.ps1 first.`);
        }
        await this.profiles.start();
        const readiness = new Promise<void>((resolve, reject) => {
            this.readyResolve = resolve;
            this.readyReject = reject;
        });
        this.provider = createWhisperProvider(this.options);
        const child = this.registerChildProcess('default', this.provider.command, this.provider.args as string[], {
            env: { ...process.env, ...this.provider.env },
            windowsHide: true,
        });
        child.stdin?.on('drain', () => {
            this.backpressured = false;
            for (const uuid of this.resetAfterDrain) this.send({ type: 'reset_peer', peerUuid: uuid });
            this.resetAfterDrain.clear();
        });
        this.startupTimer = setTimeout(() => this.failWorker(new Error('ASR startup timed out.')), this.options.workerStartupTimeoutMs);
        await readiness;
        this.log(`ASR ready; translated audio output is ${this.options.outputSampleRate} Hz PCM16 mono.`);
        if (this.diagnosticsEnabled) {
            this.log('[Diagnostic] Input statistics enabled; reporting every 10 seconds.');
            this.diagnosticsTimer = setInterval(() => this.reportInputStats(), 10000);
            this.diagnosticsTimer.unref();
        }
    }

    addPeer(uuid: string): void {
        if (!this.peers.has(uuid) && this.peers.size >= this.options.maxPeers) {
            this.warnOnce(`capacity:${uuid}`, `Peer limit (${this.options.maxPeers}) reached; ${uuid} must rejoin after capacity is available.`);
            return;
        }
        this.peers.add(uuid);
        if (this.diagnosticsEnabled && !this.inputStats.has(uuid)) {
            this.inputStats.set(uuid, { frames: 0, samples: 0, squares: 0, peak: 0, lastAt: 0 });
        }
        if (!this.profiles.snapshot(uuid)) this.warnOnce(`profile:${uuid}`, `No voice profile for ${uuid}; audio ignored until configured.`);
    }

    removePeer(uuid: string): void {
        this.peers.delete(uuid);
        this.inputStats.delete(uuid);
        this.cancelPeerJobs(uuid);
        this.send({ type: 'remove_peer', peerUuid: uuid });
        this.resetAfterDrain.delete(uuid);
        for (const key of this.warned) if (key.endsWith(`:${uuid}`)) this.warned.delete(key);
    }

    acceptAudio(uuid: string, audio: {
        samples: Int16Array; sampleRate: number; channelCount: number; bitsPerSample: number;
    }): void {
        if (this.stopped || !this.workerReady || !this.peers.has(uuid)) return;
        if (!this.profiles.snapshot(uuid)) {
            this.warnOnce(`profile:${uuid}`, `No voice profile for ${uuid}; audio ignored until configured.`);
            return;
        }
        if (!(audio.samples instanceof Int16Array) || audio.bitsPerSample !== 16 ||
            !Number.isInteger(audio.sampleRate) || audio.sampleRate < 8000 || audio.sampleRate > 96000 ||
            ![1, 2].includes(audio.channelCount) || audio.samples.length % audio.channelCount !== 0 ||
            audio.samples.length > audio.sampleRate * audio.channelCount / 4) {
            this.warnOnce(`format:${uuid}`, `Unsupported audio format from ${uuid}; expected <=250 ms of PCM16 mono/stereo.`);
            return;
        }
        if (audio.samples.length === 0) return;
        const stats = this.inputStats.get(uuid);
        if (stats) {
            if (!stats.lastAt) this.log(`[Diagnostic] First audio from ${uuid}: ${audio.sampleRate} Hz, ${audio.channelCount} channel(s), ${audio.bitsPerSample} bits.`);
            stats.frames++;
            stats.samples += audio.samples.length;
            stats.lastAt = Date.now();
            for (const sample of audio.samples) {
                stats.peak = Math.max(stats.peak, Math.abs(sample));
                stats.squares += sample * sample;
            }
        }
        if (this.backpressured) {
            this.resetAfterDrain.add(uuid);
            this.warnOnce(`backpressure:${uuid}`, `ASR input backpressure for ${uuid}; dropping live frames and resetting VAD after drain.`);
            return;
        }
        // Buffer.from(array.buffer) alone can expose samples outside this view.
        const bytes = Buffer.from(audio.samples.buffer, audio.samples.byteOffset, audio.samples.byteLength);
        if (!this.send({ type: 'audio', peerUuid: uuid, samples: bytes.toString('base64'), sampleRate: audio.sampleRate, channels: audio.channelCount })) {
            this.resetAfterDrain.add(uuid);
        }
    }

    private warnOnce(key: string, message: string): void {
        if (this.warned.has(key)) return;
        this.warned.add(key);
        this.log(message, 'warning');
    }

    private reportInputStats(): void {
        const dbfs = (amplitude: number) => amplitude > 0 ? (20 * Math.log10(amplitude / 32768)).toFixed(1) : '-inf';
        for (const [uuid, stats] of this.inputStats) {
            const rms = stats.samples ? Math.sqrt(stats.squares / stats.samples) : 0;
            const lastFrameMs = stats.lastAt ? String(Date.now() - stats.lastAt) : 'never';
            this.log(`[Diagnostic] Input ${uuid}: profile=${this.profiles.snapshot(uuid) ? 'valid' : 'missing'}, frames=${stats.frames}, peakDbFS=${dbfs(stats.peak)}, rmsDbFS=${dbfs(rms)}, lastFrameMs=${lastFrameMs}, backpressure=${this.backpressured}.`);
            stats.frames = stats.samples = stats.squares = stats.peak = 0;
        }
    }

    private diagnostic(message: string): void {
        if (this.diagnosticsEnabled) this.log(`[Diagnostic] ${message}`);
    }

    /** false means accepted with backpressure (or no longer writable), never retry that frame. */
    private send(message: Record<string, unknown>): boolean {
        if (this.stopped) return false;
        const stdin = this.childProcesses.default?.stdin;
        if (!stdin || stdin.destroyed || !stdin.writable) return false;
        if (stdin.writableLength > 2 * 1024 * 1024) {
            this.failWorker(new Error('ASR stdin exceeded its bounded buffer.'));
            return false;
        }
        try {
            const flushed = stdin.write(JSON.stringify(message) + '\n');
            if (!flushed) this.backpressured = true;
            return flushed;
        } catch (error) {
            this.failWorker(error instanceof Error ? error : new Error(String(error)));
            return false;
        }
    }

    private onWorkerMessage(value: unknown): void {
        if (this.stopped) return;
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
            this.failWorker(new Error('Invalid ASR protocol message.'));
            return;
        }
        const message = value as Record<string, unknown>;
        switch (message.type) {
            case 'ready':
                this.workerReady = true;
                clearTimeout(this.startupTimer);
                this.readyResolve?.();
                this.readyResolve = undefined;
                this.readyReject = undefined;
                break;
            case 'utterance':
                this.enqueue(message);
                break;
            case 'transcript': {
                const pending = this.pending;
                if (!pending || message.requestId !== pending.job.requestId) return;
                if (message.peerUuid !== pending.job.peerUuid || typeof message.text !== 'string' || typeof message.language !== 'string') {
                    this.failWorker(new Error('ASR transcript identity or text is invalid.'));
                    return;
                }
                clearTimeout(pending.timer);
                this.pending = undefined;
                pending.resolve({
                    requestId: pending.job.requestId, peerUuid: pending.job.peerUuid,
                    text: message.text, language: message.language,
                    languageProbability: typeof message.languageProbability === 'number' ? message.languageProbability : 0,
                });
                break;
            }
            case 'error': {
                const error = new Error(typeof message.message === 'string' ? message.message : 'ASR worker error');
                if (message.fatal === true) { this.failWorker(error); return; }
                if (this.pending && this.pending.job.requestId === message.requestId) {
                    clearTimeout(this.pending.timer);
                    this.pending.reject(error);
                    this.pending = undefined;
                } else {
                    this.log(`ASR ${String(message.peerUuid ?? '')}: ${error.message}`, 'warning');
                }
                break;
            }
            default:
                this.failWorker(new Error(`Unknown ASR message type: ${String(message.type)}`));
        }
    }

    private enqueue(message: Record<string, unknown>): void {
        const { requestId, peerUuid, endedAt, durationMs } = message;
        if (typeof requestId !== 'string' || !requestId || requestId.length > 128 ||
            typeof peerUuid !== 'string' || !peerUuid || peerUuid.length > 128 ||
            typeof endedAt !== 'number' || !Number.isFinite(endedAt) ||
            typeof durationMs !== 'number' || !Number.isFinite(durationMs) || durationMs <= 0) {
            this.failWorker(new Error('Malformed ASR utterance event.'));
            return;
        }
        if (this.active?.requestId === requestId || this.queue.some((job) => job.requestId === requestId)) {
            this.failWorker(new Error('Duplicate ASR request ID.'));
            return;
        }
        const voice = this.profiles.snapshot(peerUuid);
        if (!this.peers.has(peerUuid) || !voice || this.queue.length >= this.options.maxQueueLength) {
            this.send({ type: 'discard', requestId });
            const reason = !this.peers.has(peerUuid) ? 'peer is no longer present' :
                !voice ? 'no valid voice profile' : `queue full (${this.options.maxQueueLength} waiting sentences)`;
            this.log(`Dropped ${requestId} (${peerUuid}): ${reason}.`, 'warning');
            return;
        }
        this.queue.push({ requestId, peerUuid, endedAt, durationMs, voice, controller: new AbortController() });
        this.diagnostic(`Utterance ${requestId} (${peerUuid}): ${durationMs} ms; queued for ASR.`);
        void this.drain();
    }

    private async transcribe(job: Job): Promise<Transcript> {
        if (this.pending || !this.workerReady || this.stopped) throw new Error('ASR worker is not available.');
        return new Promise<Transcript>((resolve, reject) => {
            const timer = setTimeout(() => {
                // Do not start another GPU job while a timed-out inference may still be running.
                this.failWorker(new Error(`ASR inference timed out (${job.requestId}); worker stopped.`));
            }, this.options.requestTimeoutMs);
            this.pending = { job, resolve, reject, timer };
            this.send({ type: 'transcribe', requestId: job.requestId });
        });
    }

    private assertActive(job: Job): void {
        if (this.stopped || job.controller.signal.aborted || !this.peers.has(job.peerUuid)) {
            throw new Error('Translation canceled.');
        }
    }

    private async drain(): Promise<void> {
        if (this.processing || this.stopped) return;
        this.processing = true;
        try {
            while (this.queue.length && !this.stopped) {
                const job = this.queue.shift()!;
                this.active = job;
                const start = performance.now();
                const timings: Record<string, number> = { queueMs: Math.max(0, Date.now() - job.endedAt) };
                try {
                    this.assertActive(job);
                    this.diagnostic(`ASR started ${job.requestId}.`);
                    const transcript = await this.transcribe(job);
                    timings.asrMs = Math.round(performance.now() - start);
                    this.assertActive(job);
                    const original = transcript.text.trim();
                    this.diagnostic(`ASR completed ${job.requestId}: language=${transcript.language}, characters=${original.length}.`);
                    if (!original || !['zh', 'en'].includes(transcript.language)) {
                        this.log(`Skipped ${job.requestId}: empty transcript or unsupported language ${transcript.language}.`);
                        continue;
                    }
                    const sourceLanguage = transcript.language as Language;
                    const targetLanguage: Language = sourceLanguage === 'zh' ? 'en' : 'zh';
                    let stage = performance.now();
                    this.diagnostic(`Translation started ${job.requestId}.`);
                    // Disconnecting HTTP does not guarantee backend GPU cancellation.
                    // On peer departure, await the current request and discard its
                    // result before admitting another sentence. Shutdown still aborts.
                    const translated = await translate(original, sourceLanguage, {
                        ...this.options.ollama, timeoutMs: this.options.requestTimeoutMs,
                    }, this.lifetime.signal);
                    timings.translationMs = Math.round(performance.now() - stage);
                    this.assertActive(job);
                    stage = performance.now();
                    this.diagnostic(`TTS started ${job.requestId}.`);
                    const wav = await synthesize(translated, targetLanguage, job.voice, {
                        ...this.options.tts, timeoutMs: this.options.requestTimeoutMs,
                    }, this.lifetime.signal);
                    timings.ttsMs = Math.round(performance.now() - stage);
                    this.assertActive(job);
                    stage = performance.now();
                    const pcm = await wavToPcm(wav, {
                        ffmpegPath: this.options.ffmpegPath,
                        sampleRate: this.options.outputSampleRate,
                        timeoutMs: this.options.requestTimeoutMs,
                    }, job.controller.signal);
                    timings.convertMs = Math.round(performance.now() - stage);
                    this.assertActive(job);
                    timings.endpointToPlaybackMs = Math.max(0, Date.now() - job.endedAt);
                    this.log(JSON.stringify({
                        type: 'translation', requestId: job.requestId, peerUuid: job.peerUuid,
                        sourceLanguage, targetLanguage, original, translated, timings,
                        inputDurationMs: job.durationMs, outputDurationMs: Math.round(pcm.length / 2 / this.options.outputSampleRate * 1000),
                    }));
                    await this.play(job, pcm);
                    this.diagnostic(`Audio sent ${job.requestId}: ${pcm.length} PCM bytes on NetworkId 95; client playback is not confirmed.`);
                } catch (error) {
                    const canceled = job.controller.signal.aborted || !this.peers.has(job.peerUuid);
                    this.log(`${canceled ? 'Canceled' : 'Failed'} ${job.requestId} (${job.peerUuid}): ${error instanceof Error ? error.message : String(error)}`, canceled ? 'info' : 'warning');
                } finally {
                    this.send({ type: 'discard', requestId: job.requestId });
                    this.active = undefined;
                }
            }
        } finally {
            this.processing = false;
        }
    }

    private async play(job: Job, pcm: Buffer): Promise<void> {
        // ~100 ms packets, bounded to the existing Unity receiver's 200..16000 bytes.
        const packetBytes = Math.min(16000, Math.max(200, Math.floor(this.options.outputSampleRate / 10) * 2));
        const packets = splitPcm(pcm, packetBytes, 200);
        const header = { type: 'AudioInfo', targetPeer: '', audioLength: String(pcm.length) };
        if (Buffer.byteLength(JSON.stringify(header)) >= 100) throw new Error('Unity audio header is too large.');
        this.sendAudio(job, header, Buffer.byteLength(JSON.stringify(header)));
        let nextSendAt = performance.now();
        for (const packet of packets) {
            const waitMs = nextSendAt - performance.now();
            if (waitMs > 0) await delay(waitMs, undefined, { signal: job.controller.signal });
            this.sendAudio(job, packet, packet.length);
            // A stalled event loop must not cause a burst of overdue packets.
            nextSendAt = performance.now() + packet.length / (this.options.outputSampleRate * 2) * 1000;
        }
        const remaining = nextSendAt - performance.now();
        if (remaining > 0) await delay(remaining, undefined, { signal: job.controller.signal });
    }

    private sendAudio(job: Job, payload: Buffer | Record<string, string>, bytes: number): void {
        this.assertActive(job);
        // The shared Ubiq sender ignores socket.write() backpressure. Keep at
        // most one second of PCM in its writable queue, including framing.
        const limit = this.options.outputSampleRate * 2;
        if (!this.scene.connections.length) throw new Error('No Ubiq connection for translated audio.');
        for (const connection of this.scene.connections) {
            const socket = (connection as unknown as { socket?: {
                writableLength: number; destroyed: boolean; writable: boolean;
            } }).socket;
            if (!socket || socket.destroyed || !socket.writable || !Number.isFinite(socket.writableLength)) {
                throw new Error('Translated audio requires a writable Ubiq TCP connection.');
            }
            if (socket.writableLength + bytes + 12 > limit) {
                throw new Error('Ubiq TCP output is congested; remaining translated audio discarded.');
            }
        }
        this.scene.send(new NetworkId(95), payload);
    }

    private cancelPeerJobs(uuid: string): void {
        for (let index = this.queue.length - 1; index >= 0; index--) {
            const job = this.queue[index];
            if (job.peerUuid !== uuid) continue;
            job.controller.abort();
            this.send({ type: 'discard', requestId: job.requestId });
            this.queue.splice(index, 1);
        }
        if (this.active?.peerUuid === uuid) {
            this.active.controller.abort();
            this.send({ type: 'discard', requestId: this.active.requestId });
            // Await an in-flight ASR's terminal response before admitting another GPU job.
        }
    }

    private failWorker(error: Error): void {
        if (this.stopped) return;
        this.readyReject?.(error);
        this.readyReject = undefined;
        this.close();
        this.emit('fatal', error);
    }

    close(): void {
        if (this.stopped) return;
        this.stopped = true;
        this.lifetime.abort();
        this.workerReady = false;
        clearTimeout(this.startupTimer);
        clearInterval(this.diagnosticsTimer);
        this.inputStats.clear();
        this.readyReject?.(new Error('Translation service stopped.'));
        this.readyReject = undefined;
        if (this.pending) {
            clearTimeout(this.pending.timer);
            this.pending.reject(new Error('ASR worker stopped.'));
            this.pending = undefined;
        }
        this.active?.controller.abort();
        for (const job of this.queue) job.controller.abort();
        this.queue.length = 0;
        this.profiles.close();
        super.killAllChildProcesses();
    }

    override killAllChildProcesses(): void {
        // Also called by the base controller's signal/exit handlers.
        this.close();
    }
}
