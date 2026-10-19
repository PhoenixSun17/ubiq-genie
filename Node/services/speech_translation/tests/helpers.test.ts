import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, rename, rm, writeFile } from 'node:fs/promises';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test, type TestContext } from 'node:test';
import { splitPcm, wavToPcm } from '../audio';
import { localEndpoint, synthesize, translate } from '../clients';
import { VoiceProfileStore, wavDuration } from '../profiles';
import { JsonLineDecoder } from '../protocol';
import type { VoiceProfile } from '../types';

const UUID_A = '12345678-1234-4123-8123-123456789abc';
const UUID_B = 'abcdef01-1234-4123-8123-123456789abc';
const signal = () => new AbortController().signal;

function pcmWav(seconds: number, sampleRate = 16000, channels = 1): Buffer {
    const dataBytes = Math.round(seconds * sampleRate) * channels * 2;
    const wav = Buffer.alloc(44 + dataBytes);
    wav.write('RIFF', 0); wav.writeUInt32LE(36 + dataBytes, 4); wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(channels, 22);
    wav.writeUInt32LE(sampleRate, 24); wav.writeUInt32LE(sampleRate * channels * 2, 28);
    wav.writeUInt16LE(channels * 2, 32); wav.writeUInt16LE(16, 34);
    wav.write('data', 36); wav.writeUInt32LE(dataBytes, 40);
    for (let offset = 44; offset < wav.length; offset += 2) wav.writeInt16LE(1234, offset);
    return wav;
}

async function localServer(t: TestContext, handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<string> {
    const server = createServer(handler);
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    t.after(async () => {
        server.closeAllConnections();
        await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    });
    const address = server.address();
    assert.ok(address && typeof address !== 'string');
    return `http://127.0.0.1:${address.port}`;
}

async function readRequest(req: IncomingMessage): Promise<Record<string, any>> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, any>;
}

test('NDJSON handles split UTF-8, coalesced records, CRLF, and a final line without LF', () => {
    const messages: unknown[] = [];
    const errors: Error[] = [];
    const decoder = new JsonLineDecoder((message) => messages.push(message), (error) => errors.push(error));
    const bytes = Buffer.from('{"text":"你好"}\r\n{"type":"ready"}\n{"last":true}');
    for (const byte of bytes) decoder.push(Buffer.from([byte]));
    decoder.end();
    assert.deepEqual(messages, [{ text: '你好' }, { type: 'ready' }, { last: true }]);
    assert.deepEqual(errors, []);
    assert.throws(() => decoder.push('{}\n'), /ended/);
});

test('NDJSON rejects oversized/malformed lines and resumes at the next record', () => {
    const messages: unknown[] = [];
    const errors: Error[] = [];
    const decoder = new JsonLineDecoder((message) => messages.push(message), (error) => errors.push(error), 20);
    decoder.push('x'.repeat(21));
    decoder.push('x'.repeat(40) + '\ninvalid\n{"ok":true}\n');
    decoder.end();
    assert.equal(errors.length, 2);
    assert.deepEqual(messages, [{ ok: true }]);
});

test('PCM preserves every byte and rebalances a tiny final packet above the Unity header threshold', () => {
    for (const length of [200, 16000, 16002, 16198, 16200, 32002, 48000, 3840000]) {
        const input = Buffer.alloc(length, 0x36);
        const chunks = splitPcm(input);
        assert.ok(chunks.every((chunk) => chunk.length >= 200 && chunk.length <= 16000 && chunk.length % 2 === 0));
        assert.deepEqual(Buffer.concat(chunks), input);
    }
    assert.deepEqual(splitPcm(Buffer.alloc(16002)).map((chunk) => chunk.length), [15802, 200]);
    assert.throws(() => splitPcm(Buffer.alloc(198)), /minimum/);
    assert.throws(() => splitPcm(Buffer.alloc(201)), /even/);
    assert.throws(() => splitPcm(Buffer.alloc(602), 400, 351), /even/);
    assert.throws(() => splitPcm(Buffer.alloc(402), 400, 300), /cannot fit/);
});

test('WAV duration validates channel/rate metadata and complete sample data', () => {
    assert.equal(wavDuration(pcmWav(3)), 3);
    assert.equal(wavDuration(pcmWav(10, 48000, 2)), 10);
    assert.throws(() => wavDuration(pcmWav(3).subarray(0, 50)), /truncated/);
    const wrongRate = pcmWav(3);
    wrongRate.writeUInt32LE(1, 28);
    assert.throws(() => wavDuration(wrongRate), /format/);
});

test('profile replacement validates every user and retains last good profiles after failure', async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'speech-profile-test-'));
    const path = join(directory, 'profiles.json');
    await writeFile(join(directory, 'a.wav'), pcmWav(3));
    await writeFile(join(directory, 'b.wav'), pcmWav(4));
    const a: VoiceProfile = { refAudioPath: 'a.wav', promptText: '第一位用户', promptLanguage: 'zh' };
    const b: VoiceProfile = { refAudioPath: 'b.wav', promptText: 'Second speaker', promptLanguage: 'en' };
    await writeFile(path, JSON.stringify({ [UUID_A]: a, [UUID_B]: b }));
    const errors: string[] = [];
    const store = new VoiceProfileStore(path, (message) => errors.push(message));
    t.after(async () => { store.close(); await rm(directory, { recursive: true, force: true }); });
    const changes: string[][] = [];
    store.on('changed', (uuids: string[]) => changes.push(uuids));
    assert.equal(await store.reload(), true);
    assert.equal(store.snapshot(UUID_A)?.refAudioPath, join(directory, 'a.wav'));
    assert.equal(store.snapshot(UUID_B)?.promptText, 'Second speaker');
    const snapshot = store.snapshot(UUID_A)!;
    snapshot.promptText = 'Must not mutate live config';
    assert.equal(store.snapshot(UUID_A)?.promptText, a.promptText);
    await writeFile(path, JSON.stringify({ [UUID_A]: { ...a, promptText: 'Changed' }, [UUID_B]: { ...b, refAudioPath: 'missing.wav' } }));
    assert.equal(await store.reload(), false);
    assert.equal(store.snapshot(UUID_A)?.promptText, a.promptText);
    assert.equal(changes.length, 1);
    assert.ok(errors.some((message) => message.includes('keeping last valid')));
    await writeFile(path, JSON.stringify({ [UUID_B]: b }));
    assert.equal(await store.reload(), true);
    assert.deepEqual(changes[1], [UUID_A]);
    assert.equal(store.snapshot(UUID_A), undefined);
    await writeFile(path, '{}');
    assert.equal(await store.reload(), true);
    assert.equal(store.snapshot(UUID_B), undefined);
});

test('profile validation rejects unsupported language, invalid UUID, and duration outside 3..10 seconds', async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'speech-profile-invalid-'));
    const path = join(directory, 'profiles.json');
    await writeFile(join(directory, 'short.wav'), pcmWav(2));
    await writeFile(join(directory, 'long.wav'), pcmWav(11));
    const store = new VoiceProfileStore(path, () => {});
    t.after(async () => { store.close(); await rm(directory, { recursive: true, force: true }); });
    for (const invalid of [
        { [UUID_A]: { refAudioPath: 'short.wav', promptText: 'hello', promptLanguage: 'en' } },
        { [UUID_A]: { refAudioPath: 'long.wav', promptText: 'hello', promptLanguage: 'en' } },
        { [UUID_A]: { refAudioPath: 'short.wav', promptText: 'hello', promptLanguage: 'fr' } },
        { 'not-a-uuid': { refAudioPath: 'short.wav', promptText: 'hello', promptLanguage: 'en' } },
    ]) {
        await writeFile(path, JSON.stringify(invalid));
        assert.equal(await store.reload(), false);
    }
});

test('profile watcher observes atomic editor replacements', async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'speech-profile-watch-'));
    const path = join(directory, 'profiles.json');
    await writeFile(path, '{}');
    await writeFile(join(directory, 'voice.wav'), pcmWav(3));
    const store = new VoiceProfileStore(path, () => {});
    t.after(async () => { store.close(); await rm(directory, { recursive: true, force: true }); });
    await store.start();
    const changed = once(store, 'changed', { signal: AbortSignal.timeout(3000) });
    const replacement = join(directory, 'profiles.tmp');
    await writeFile(replacement, JSON.stringify({ [UUID_A]: {
        refAudioPath: 'voice.wav', promptText: 'Reference', promptLanguage: 'en',
    } }));
    await rename(replacement, path);
    const [uuids] = await changed;
    assert.deepEqual(uuids, [UUID_A]);
    assert.equal(store.snapshot(UUID_A)?.promptText, 'Reference');
});

test('local service origins reject remote destinations, credentials, and embedded paths', () => {
    for (const invalid of ['https://127.0.0.1:9880', 'http://example.com', 'http://user@localhost', 'http://localhost/tts', 'http://localhost?q=1']) {
        assert.throws(() => localEndpoint(invalid, '/tts'), /loopback/);
    }
    assert.equal(localEndpoint('http://127.0.0.1:9880', '/tts').href, 'http://127.0.0.1:9880/tts');
});

test('Ollama requests are independent, disable thinking, and only use message.content', async (t) => {
    const bodies: Record<string, any>[] = [];
    const baseUrl = await localServer(t, (req, res) => {
        void readRequest(req).then((body) => {
            bodies.push(body);
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ done: true, message: { content: bodies.length === 1 ? 'Hello.' : '你好。', thinking: 'Do not speak this' } }));
        });
    });
    const options = { baseUrl, model: 'qwen3:4b', timeoutMs: 1000 };
    assert.equal(await translate('你好。', 'zh', options, signal()), 'Hello.');
    assert.equal(await translate('Hello.', 'en', options, signal()), '你好。');
    for (const body of bodies) {
        assert.equal(body.think, false); assert.equal(body.stream, false);
        assert.deepEqual(body.options, { temperature: 0, num_ctx: 4096 });
        assert.equal(body.messages.length, 2);
    }
    assert.equal(bodies[1].messages[1].content, 'Hello.');
    assert.match(bodies[1].messages[0].content, /into Chinese/);
});

test('GPT-SoVITS receives the selected reference per sentence and returns full WAV', async (t) => {
    const wav = pcmWav(1);
    let received: Record<string, any> | undefined;
    const baseUrl = await localServer(t, (req, res) => {
        void readRequest(req).then((body) => { received = body; res.setHeader('Content-Type', 'audio/wav'); res.end(wav); });
    });
    const profile: VoiceProfile = { refAudioPath: 'C:\\voice.wav', promptText: 'Sample voice', promptLanguage: 'en' };
    assert.deepEqual(await synthesize('你好', 'zh', profile, { baseUrl, timeoutMs: 1000 }, signal()), wav);
    assert.equal(received?.ref_audio_path, profile.refAudioPath);
    assert.equal(received?.prompt_text, profile.promptText);
    assert.equal(received?.text_lang, 'zh');
    assert.equal(received?.batch_size, 1);
    assert.equal(received?.media_type, 'wav');
    assert.equal(received?.streaming_mode, false);
});

test('HTTP errors, oversized responses, invalid translation, timeout, and cancellation reject', async (t) => {
    let mode = 'status';
    const baseUrl = await localServer(t, (_req, res) => {
        if (mode === 'timeout') return;
        if (mode === 'status') { res.statusCode = 503; res.end('Unavailable'); return; }
        if (mode === 'redirect') { res.statusCode = 302; res.setHeader('Location', 'http://example.com'); res.end(); return; }
        if (mode === 'oversize') { res.setHeader('Content-Length', 1024 * 1024 + 1); res.flushHeaders(); return; }
        if (mode === 'truncated') { res.end(JSON.stringify({ done: true, done_reason: 'length', message: { content: 'cut off' } })); return; }
        res.end(JSON.stringify({ done: true, message: { content: '' } }));
    });
    const options = { baseUrl, model: 'qwen3:4b', timeoutMs: 1000 };
    await assert.rejects(translate('hello', 'en', options, signal()), /HTTP 503/);
    mode = 'redirect';
    await assert.rejects(translate('hello', 'en', options, signal()), /HTTP 302/);
    mode = 'oversize';
    await assert.rejects(translate('hello', 'en', options, signal()), /exceeds/);
    mode = 'invalid';
    await assert.rejects(translate('hello', 'en', options, signal()), /Text must contain/);
    mode = 'truncated';
    await assert.rejects(translate('hello', 'en', options, signal()), /truncated/);
    mode = 'timeout';
    await assert.rejects(translate('hello', 'en', { ...options, timeoutMs: 30 }, signal()), /timed out/);
    const controller = new AbortController();
    const request = translate('hello', 'en', options, controller.signal);
    controller.abort();
    await assert.rejects(request, /cancelled/);
});

test('cancelled conversion never starts FFmpeg', async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(wavToPcm(pcmWav(1), {
        ffmpegPath: 'this-program-must-not-run', sampleRate: 16000, timeoutMs: 1000,
    }, controller.signal), /cancelled/);
});

// Opt-in after approval; uses the installed project-local FFmpeg, no model/GPU/Ubiq.
test('FFmpeg converts stereo 48 kHz WAV to mono PCM16 at the configured 16 kHz rate', {
    skip: !process.env.SPEECH_TRANSLATION_TEST_FFMPEG,
}, async () => {
    const pcm = await wavToPcm(pcmWav(1, 48000, 2), {
        ffmpegPath: process.env.SPEECH_TRANSLATION_TEST_FFMPEG!, sampleRate: 16000, timeoutMs: 10000,
    }, signal());
    assert.equal(pcm.length, 16000 * 2);
    assert.ok(Math.abs(pcm.readInt16LE(16000) - 1234) <= 2);
    assert.deepEqual(Buffer.concat(splitPcm(pcm)), pcm);
});
