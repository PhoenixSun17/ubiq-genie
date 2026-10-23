import assert from 'node:assert/strict';
import { EventEmitter, once } from 'node:events';
import { createServer, type ServerResponse } from 'node:http';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setImmediate as nextTurn } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { test, type TestContext } from 'node:test';
import type { NetworkScene } from '@ucl-vr/ubiq';
import { readTranslationConfig } from '../config';
import { SpeechTranslationService } from '../service';
import { GptSovitsTranslation } from '../../../apps/gpt_sovits_translation/app';
import type { VoiceProfile } from '../types';
import { MotionReplayController, type MotionReplayLifecycle } from '../motion_replay';

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
    const lifecycle: MotionReplayLifecycle[] = [];
    service.on('motionReplay', (event: MotionReplayLifecycle) => lifecycle.push(event));
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
    return { service, internal, profiles, commands, broadcasts, logs, emit, utterance, transcript, requested, socket, scene, lifecycle };
}

/** Load real validated profiles while retaining the in-memory model and network transports. */
async function catalogHarness(t: TestContext) {
    const h = harness(t);
    const directory = await mkdtemp(join(tmpdir(), 'speech-scheduler-catalog-'));
    const path = join(directory, 'profiles.json');
    const wav = Buffer.alloc(44 + 3 * 16000 * 2);
    wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32);
    wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
    await writeFile(join(directory, 'a.wav'), wav);
    await writeFile(join(directory, 'b.wav'), wav);
    const mapping = Object.fromEntries(h.profiles);
    await writeFile(path, JSON.stringify(mapping));
    delete h.internal.profiles.snapshot;
    h.internal.profiles.path = path;
    assert.equal(await h.internal.profiles.reload(), true);
    t.after(async () => { await rm(directory, { recursive: true, force: true }); });
    h.commands.length = 0;
    return { ...h, path, mapping };
}

test('scheduler preserves sentence order and rejects the newest sentence beyond eight waiting jobs', async (t) => {
    const h = harness(t);
    for (let i = 0; i < 10; i++) h.utterance(`job-${i}`, i % 2 ? B : A);
    assert.deepEqual(h.requested(), ['job-0']);
    assert.ok(h.commands.some((command) => command.type === 'discard' && command.requestId === 'job-9'));
    assert.deepEqual(h.lifecycle.map((event) => event.requestId), Array.from({ length: 9 }, (_, i) => `job-${i}`));
    for (let i = 0; i < 9; i++) {
        assert.equal(h.requested().at(-1), `job-${i}`);
        h.transcript(`job-${i}`, i % 2 ? B : A);
        await nextTurn();
    }
    assert.deepEqual(h.requested(), Array.from({ length: 9 }, (_, i) => `job-${i}`));
    assert.equal(h.broadcasts.length, 0);
    assert.equal(h.internal.processing, false);
    assert.equal(h.lifecycle.filter((event) => event.type === 'terminal' && event.status === 'skipped').length, 9);
});

test('leaving cancels that user only and a late ASR response cannot translate or play', async (t) => {
    const h = harness(t);
    h.utterance('active-a');
    h.utterance('waiting-a');
    h.utterance('waiting-b', B);
    h.service.removePeer(A);
    assert.deepEqual(h.lifecycle.filter((event) => event.type === 'terminal').map((event) => event.requestId),
        ['waiting-a', 'active-a'], 'cancellation reaches clients without awaiting inference');
    assert.deepEqual(h.requested(), ['active-a'], 'the next ASR must wait for the active native call to finish');
    h.transcript('active-a', A, '迟到的结果', 'zh');
    await nextTurn();
    assert.deepEqual(h.requested(), ['active-a', 'waiting-b']);
    assert.equal(h.broadcasts.length, 0);
    assert.ok(h.logs.some((message) => message.includes('Canceled active-a')));
    h.transcript('waiting-b', B);
    await nextTurn();
    assert.equal(h.internal.processing, false);
    assert.equal(h.lifecycle.filter((event) => event.type === 'terminal' && event.requestId === 'active-a').length, 1);
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

test('service binding resets only the changed peer and preserves active and queued voice snapshots', async (t) => {
    const h = await catalogHarness(t);
    const oldId = h.service.getBoundVoiceId(A)!;
    const nextId = h.service.getBoundVoiceId(B)!;
    h.utterance('active-a');
    h.utterance('old-a');
    const active = h.internal.active;
    const queued = h.internal.queue[0];
    h.service.bindVoice(A, nextId);
    h.utterance('new-a');
    assert.equal(h.service.getBoundVoiceId(A), nextId);
    assert.equal(h.service.getBoundVoiceId(B), nextId);
    assert.equal(active.voiceId, oldId);
    assert.equal(queued.voiceId, oldId);
    assert.equal(h.internal.queue[1].voiceId, nextId);
    assert.equal(active.controller.signal.aborted, false);
    assert.equal(queued.controller.signal.aborted, false);
    assert.deepEqual(h.commands.filter((command) => command.type === 'reset_peer').map((command) => command.peerUuid), [A]);
    h.service.bindVoice(A, nextId);
    assert.equal(h.commands.filter((command) => command.type === 'reset_peer').length, 1);
    h.transcript('active-a');
    await nextTurn();
    h.transcript('old-a');
    await nextTurn();
    h.transcript('new-a');
    await nextTurn();
    assert.deepEqual(h.requested(), ['active-a', 'old-a', 'new-a']);
});

test('deleting a catalog voice cancels its snapshots even after the peer selected another voice', async (t) => {
    const h = await catalogHarness(t);
    const nextId = h.service.getBoundVoiceId(B)!;
    h.utterance('active-old-a');
    h.utterance('queued-old-a');
    const active = h.internal.active;
    h.service.bindVoice(A, nextId);
    h.utterance('queued-new-a');
    h.utterance('queued-b', B);
    let catalogEvents = 0;
    h.service.on('voiceCatalogChanged', () => { catalogEvents++; });
    await writeFile(h.path, JSON.stringify({ [B]: h.mapping[B] }));
    assert.equal(await h.internal.profiles.reload(), true);
    assert.equal(catalogEvents, 1);
    assert.equal(active.controller.signal.aborted, true);
    assert.deepEqual(h.internal.queue.map((job: { requestId: string }) => job.requestId), ['queued-new-a', 'queued-b']);
    assert.equal(h.service.getBoundVoiceId(A), nextId, 'the new selected voice remains available');
    assert.ok(h.commands.some((command) => command.type === 'discard' && command.requestId === 'queued-old-a'));
    h.transcript('active-old-a', A, 'Late transcript');
    await nextTurn();
    assert.deepEqual(h.requested(), ['active-old-a', 'queued-new-a']);
    assert.equal(h.broadcasts.length, 0);
    h.transcript('queued-new-a');
    await nextTurn();
    h.transcript('queued-b', B);
    await nextTurn();
});

test('missing selected voices reset capture and cancel jobs without falling back to legacy profiles', async (t) => {
    const h = await catalogHarness(t);
    h.service.bindVoice(A, h.service.getBoundVoiceId(B)!);
    h.utterance('active-selected');
    h.utterance('queued-selected');
    const active = h.internal.active;
    h.commands.length = 0;
    await writeFile(h.path, JSON.stringify({ [A]: h.mapping[A] }));
    assert.equal(await h.internal.profiles.reload(), true);
    assert.equal(h.service.getBoundVoiceId(A), undefined);
    assert.equal(h.internal.profiles.snapshot(A), undefined);
    assert.equal(active.controller.signal.aborted, true);
    assert.equal(h.internal.queue.length, 0);
    assert.ok(h.commands.some((command) => command.type === 'reset_peer' && command.peerUuid === A));
    h.service.acceptAudio(A, { samples: new Int16Array([1, 2]), sampleRate: 16000, channelCount: 1, bitsPerSample: 16 });
    assert.equal(h.commands.filter((command) => command.type === 'audio').length, 0);
    h.transcript('active-selected', A, 'Late transcript');
    await nextTurn();
    assert.equal(h.broadcasts.length, 0);
});

test('binding rejects unavailable peers, IDs and lifecycle states; leaving clears session overrides', async (t) => {
    const h = await catalogHarness(t);
    const aId = h.service.getBoundVoiceId(A)!;
    const bId = h.service.getBoundVoiceId(B)!;
    assert.throws(() => h.service.bindVoice('unknown-peer', aId), /no longer in this room/);
    assert.throws(() => h.service.bindVoice(A, 'invalid'), /Invalid voice selection/);
    assert.throws(() => h.service.bindVoice(A, '0'.repeat(64)), /no longer available/);
    h.internal.workerReady = false;
    assert.throws(() => h.service.bindVoice(A, bId), /not ready/);
    h.internal.workerReady = true;
    h.service.bindVoice(A, bId);
    assert.equal(h.service.getBoundVoiceId(A), bId);
    h.service.removePeer(A);
    assert.throws(() => h.service.bindVoice(A, bId), /no longer in this room/);
    h.service.addPeer(A);
    assert.equal(h.service.getBoundVoiceId(A), aId);
    h.service.close();
    assert.throws(() => h.service.bindVoice(A, bId), /not ready/);
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
    assert.deepEqual(h.lifecycle.filter((event) => event.type === 'terminal').map((event) =>
        [event.requestId, event.type === 'terminal' ? event.status : '']), [
        ['failed', 'failed'], ['unsupported', 'skipped'], ['empty', 'skipped'],
    ]);
});

test('accepted motion timing uses a monotonic received interval despite worker wall-clock changes', async (t) => {
    const h = harness(t);
    h.emit({ type: 'utterance', requestId: 'clock', peerUuid: A, endedAt: 9e12, durationMs: 14500 });
    const event = h.lifecycle[0];
    assert.equal(event.type, 'accepted');
    if (event.type !== 'accepted') return;
    assert.equal(event.speakerUuid, A);
    assert.equal(event.sourceEndMs - event.sourceStartMs, 14500);
    assert.ok(event.sourceEndMs < 9e12);
    h.transcript('clock');
    await nextTurn();
});

test('shutdown publishes one terminal for each pending clip before closing the bridge', async (t) => {
    const h = harness(t);
    h.utterance('active');
    h.utterance('waiting', B);
    let terminalCountAtClose = 0;
    h.service.on('translationClosed', () => {
        terminalCountAtClose = h.lifecycle.filter((event) => event.type === 'terminal').length;
    });
    h.service.close();
    await nextTurn();
    assert.equal(terminalCountAtClose, 2);
    assert.equal(h.lifecycle.filter((event) => event.type === 'terminal').length, 2);
    assert.ok(h.lifecycle.filter((event) => event.type === 'terminal').every((event) =>
        event.type === 'terminal' && event.status === 'cancelled'));
});

test('bridge writes play metadata immediately before unchanged ID 95 audio and blocks unassociated audio', async (t) => {
    const h = harness(t);
    const writes: { id: number; payload: unknown }[] = [];
    const scene = Object.assign(h.scene, {
        register: () => undefined,
        unregister: () => undefined,
        send: (id: unknown, payload: unknown) => {
            writes.push({ id: (id as { b: number }).b, payload });
        },
    });
    const room = Object.assign(new EventEmitter(), {
        peer: { uuid: B, setProperty: () => undefined },
        peers: new Map([[A, { uuid: A }]]),
        room: { uuid: '6765c52b-3ad6-4fb0-9030-2c9a05dc4731' },
    });
    const bridge = new MotionReplayController(scene as unknown as NetworkScene, room, h.service, 16000, () => undefined);
    t.after(() => bridge.close());
    const job = { requestId: 'audio-1', peerUuid: A, controller: new AbortController() };
    await h.internal.play(job, Buffer.alloc(200));
    assert.deepEqual(writes.map((write) => write.id), [101, 95, 95]);
    assert.deepEqual(writes[1].payload, { type: 'AudioInfo', targetPeer: '', audioLength: '200' });
    assert.ok(Buffer.isBuffer(writes[2].payload));
    assert.equal((writes[0].payload as { sampleCount: number }).sampleCount, 100);
    writes.length = 0;
    h.socket.writableLength = 32000;
    await assert.rejects(h.internal.play(job, Buffer.alloc(200)), /congested/);
    assert.equal(writes.length, 0);
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
