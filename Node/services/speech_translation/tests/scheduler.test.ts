import assert from 'node:assert/strict';
import { EventEmitter, once } from 'node:events';
import { createServer, type ServerResponse } from 'node:http';
import { setImmediate as nextTurn } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { test, type TestContext } from 'node:test';
import type { NetworkScene } from 'ubiq-server/ubiq';
import { readTranslationConfig } from '../config';
import { SpeechTranslationService } from '../service';
import { GptSovitsTranslation } from '../../../apps/gpt_sovits_translation/app';
import type { VoiceProfile } from '../types';

const A = '12345678-1234-4123-8123-123456789abc';
const B = 'abcdef01-1234-4123-8123-123456789abc';

/** In-memory transport only: no child process, model, HTTP server or Ubiq connection. */
function harness(t: TestContext) {
    const options = readTranslationConfig(fileURLToPath(new URL('../../../apps/gpt_sovits_translation/config.json', import.meta.url)));
    const commands: Record<string, any>[] = [];
    const broadcasts: unknown[] = [];
    const logs: string[] = [];
    const events = ['exit', 'SIGINT', 'SIGTERM'] as const;
    const processEvents: EventEmitter = process;
    const previousListeners = new Map(events.map((event) => [event, processEvents.listeners(event)]));
    const socket = { writableLength: 0, destroyed: false, writable: true };
    const scene = { connections: [{ socket }], getComponent: () => ({}),
        send: (_id: unknown, message: unknown) => { broadcasts.push(message); } };
    const service = new SpeechTranslationService(scene as unknown as NetworkScene, options);
    // The test adapter replaces only I/O. Scheduling, framing and cancellation run unchanged.
    const internal = service as any;
    const profiles = new Map<string, VoiceProfile>([
        [A, { refAudioPath: 'a.wav', promptText: 'Speaker A', promptLanguage: 'en' }],
        [B, { refAudioPath: 'b.wav', promptText: 'Speaker B', promptLanguage: 'en' }],
    ]);
    internal.profiles.snapshot = (uuid: string) => {
        const profile = profiles.get(uuid);
        return profile ? { ...profile } : undefined;
    };
    service.log = (message: string) => { logs.push(message); };
    internal.childProcesses.default = {
        stdin: { writable: true, destroyed: false, writableLength: 0,
            write: (line: string) => { commands.push(JSON.parse(line)); return true; } },
        kill: () => {},
    };
    const emit = (message: Record<string, unknown>) => service.emit('data', Buffer.from(JSON.stringify(message) + '\n'));
    emit({ type: 'ready' });
    service.addPeer(A);
    service.addPeer(B);
    t.after(() => {
        service.close();
        for (const event of events) {
            for (const listener of processEvents.listeners(event)) {
                if (!previousListeners.get(event)!.includes(listener)) processEvents.removeListener(event, listener as (...args: any[]) => void);
            }
        }
    });
    const utterance = (requestId: string, peerUuid = A) => emit({ type: 'utterance', requestId, peerUuid, endedAt: Date.now(), durationMs: 1000 });
    const transcript = (requestId: string, peerUuid = A, text = '', language = 'zh') => emit({ type: 'transcript', requestId, peerUuid, text, language });
    const requested = () => commands.filter((command) => command.type === 'transcribe').map((command) => command.requestId);
    return { service, internal, profiles, commands, broadcasts, logs, emit, utterance, transcript, requested, socket, scene };
}

test('scheduler preserves sentence order and rejects the newest sentence beyond eight waiting jobs', async (t) => {
    const h = harness(t);
    for (let i = 0; i < 10; i++) h.utterance(`job-${i}`, i % 2 ? B : A);
    assert.deepEqual(h.requested(), ['job-0']);
    assert.ok(h.commands.some((command) => command.type === 'discard' && command.requestId === 'job-9'));
    for (let i = 0; i < 9; i++) {
        assert.equal(h.requested().at(-1), `job-${i}`);
        h.transcript(`job-${i}`, i % 2 ? B : A);
        await nextTurn();
    }
    assert.deepEqual(h.requested(), Array.from({ length: 9 }, (_, i) => `job-${i}`));
    assert.equal(h.broadcasts.length, 0);
    assert.equal(h.internal.processing, false);
});

test('leaving cancels that user only and a late ASR response cannot translate or play', async (t) => {
    const h = harness(t);
    h.utterance('active-a');
    h.utterance('waiting-a');
    h.utterance('waiting-b', B);
    h.service.removePeer(A);
    assert.deepEqual(h.requested(), ['active-a'], 'the next ASR must wait for the active native call to finish');
    h.transcript('active-a', A, '迟到的结果', 'zh');
    await nextTurn();
    assert.deepEqual(h.requested(), ['active-a', 'waiting-b']);
    assert.equal(h.broadcasts.length, 0);
    assert.ok(h.logs.some((message) => message.includes('Canceled active-a')));
    h.transcript('waiting-b', B);
    await nextTurn();
    assert.equal(h.internal.processing, false);
});

test('voice updates keep queued snapshots and removing a profile cancels its pending sentences', async (t) => {
    const h = harness(t);
    h.utterance('active-b', B);
    h.utterance('old-a');
    h.profiles.set(A, { refAudioPath: 'new-a.wav', promptText: 'New voice', promptLanguage: 'en' });
    h.internal.profiles.emit('changed', [A]);
    h.utterance('new-a');
    assert.equal(h.internal.queue[0].voice.refAudioPath, 'a.wav');
    assert.equal(h.internal.queue[1].voice.refAudioPath, 'new-a.wav');
    h.profiles.delete(A);
    h.internal.profiles.emit('changed', [A]);
    h.transcript('active-b', B);
    await nextTurn();
    assert.deepEqual(h.requested(), ['active-b']);
    for (const id of ['old-a', 'new-a']) {
        assert.ok(h.commands.some((command) => command.type === 'discard' && command.requestId === id));
    }
});

test('an ASR failure or unsupported language does not block the next sentence', async (t) => {
    const h = harness(t);
    h.utterance('failed');
    h.utterance('unsupported', B);
    h.utterance('empty');
    h.emit({ type: 'error', requestId: 'failed', peerUuid: A, message: 'Synthetic ASR failure' });
    await nextTurn();
    h.transcript('unsupported', B, 'bonjour', 'fr');
    await nextTurn();
    h.transcript('empty');
    await nextTurn();
    assert.deepEqual(h.requested(), ['failed', 'unsupported', 'empty']);
    assert.equal(h.broadcasts.length, 0);
    assert.ok(h.logs.some((message) => message.includes('Synthetic ASR failure')));
    assert.equal(h.internal.processing, false);
});

test('audio uses only the typed-array view and unconfigured or departed users are ignored', (t) => {
    const h = harness(t);
    const allocation = new Int16Array([30000, 10, -20, 30000]);
    const audio = { samples: allocation.subarray(1, 3), sampleRate: 16000, channelCount: 1, bitsPerSample: 16 };
    h.service.acceptAudio(A, audio);
    const message = h.commands.find((command) => command.type === 'audio')!;
    assert.equal(message.peerUuid, A);
    assert.deepEqual(Buffer.from(message.samples, 'base64'), Buffer.from([10, 0, 236, 255]));
    h.profiles.delete(B);
    h.service.acceptAudio(B, audio);
    h.service.removePeer(A);
    h.service.acceptAudio(A, audio);
    assert.equal(h.commands.filter((command) => command.type === 'audio').length, 1);
});

test('app excludes its own UUID and releases ASR when the room rejects joining', (t) => {
    // Skip the application constructor so no real NetworkScene is created.
    const app = Object.create(GptSovitsTranslation.prototype) as any;
    const room = Object.assign(new EventEmitter(), { peer: { uuid: 'local-node' }, peers: new Map([
        ['local-node', { uuid: 'local-node' }], [A, { uuid: A }],
    ]) });
    const added: string[] = [];
    let closed = false;
    const service = Object.assign(new EventEmitter(), {
        addPeer: (uuid: string) => { added.push(uuid); },
        close: () => { closed = true; },
    });
    app.roomClient = room;
    app.translation = service;
    app.components = { voipReceiver: new EventEmitter() };
    const logs: string[] = [];
    app.log = (message: string) => { logs.push(message); };
    app.definePipeline();
    room.emit('OnPeerAdded', { uuid: 'local-node' });
    room.emit('OnPeerAdded', { uuid: B });
    assert.deepEqual(added, [A, B]);
    const interceptedExit = new Error('Intercepted process exit');
    t.mock.method(process, 'exit', (code?: string | number | null): never => {
        assert.equal(code, 1);
        throw interceptedExit;
    });
    assert.throws(() => room.emit('OnJoinRejected', 'Room full'), (error) => error === interceptedExit);
    assert.equal(closed, true);
    assert.ok(logs.some((message) => message.includes('Room full')));
});

test('congested TCP output stops audio before its writable queue grows beyond one second', async (t) => {
    const h = harness(t);
    const job = { peerUuid: A, controller: new AbortController() };
    h.socket.writableLength = 32000;
    await assert.rejects(h.internal.play(job, Buffer.alloc(6400)), /congested/);
    assert.equal(h.broadcasts.length, 0, 'a congested connection must not receive another header');
    h.socket.writableLength = 0;
    h.scene.send = (_id, payload) => {
        h.broadcasts.push(payload);
        if (Buffer.isBuffer(payload)) h.socket.writableLength = 32000;
    };
    await assert.rejects(h.internal.play(job, Buffer.alloc(6400)), /congested/);
    assert.equal(h.broadcasts.length, 2, 'stop after the first packet when the socket stops draining');
    assert.ok(Buffer.isBuffer(h.broadcasts[1]));
    assert.equal((h.broadcasts[1] as Buffer).length, 3200);
});

test('peer departure waits for the current backend response and discards it before the next ASR', { timeout: 3000 }, async (t) => {
    const h = harness(t);
    let requestArrived!: (response: ServerResponse) => void;
    const responseReady = new Promise<ServerResponse>((resolve) => { requestArrived = resolve; });
    const server = createServer((_req, res) => { requestArrived(res); });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    t.after(() => { server.closeAllConnections(); server.close(); });
    const address = server.address();
    assert.ok(address && typeof address !== 'string');
    h.internal.options.ollama.baseUrl = `http://127.0.0.1:${address.port}`;
    h.utterance('a');
    h.transcript('a', A, '你好', 'zh');
    const response = await responseReady;
    h.utterance('b', B);
    h.service.removePeer(A);
    await nextTurn();
    assert.deepEqual(h.requested(), ['a']);
    assert.equal(response.destroyed, false);
    const nextAsr = new Promise<void>((resolve) => {
        const write = h.internal.childProcesses.default.stdin.write;
        h.internal.childProcesses.default.stdin.write = (line: string) => {
            const result = write(line);
            if (JSON.parse(line).type === 'transcribe') resolve();
            return result;
        };
    });
    response.end(JSON.stringify({ done: true, message: { content: 'Hello' } }));
    await nextAsr;
    assert.deepEqual(h.requested(), ['a', 'b']);
    assert.equal(h.broadcasts.length, 0);
    h.transcript('b', B);
    await nextTurn();
});
