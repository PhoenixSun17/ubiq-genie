import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test, type TestContext } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import type { NetworkScene } from '@ucl-vr/ubiq';
import {
    MotionReplayController, MOTION_REPLAY_NETWORK_ID, MOTION_REPLAY_PROPERTY,
    type MotionReplayLifecycle,
} from '../motion_replay';

const BACKEND = '12345678-1234-4123-8123-123456789abc';
const SPEAKER = 'abcdef01-1234-4123-8123-123456789abc';
const ROOM = '6765c52b-3ad6-4fb0-9030-2c9a05dc4731';
const accepted: MotionReplayLifecycle = {
    type: 'accepted', requestId: 'request-1', speakerUuid: SPEAKER, sourceStartMs: 100, sourceEndMs: 1100,
};
const play: MotionReplayLifecycle = {
    type: 'play', requestId: 'request-1', speakerUuid: SPEAKER, audioLength: 32000, sampleRate: 16000, sampleCount: 16000,
};

function harness(t: TestContext, joined = true) {
    const properties = new Map<string, string>();
    const service = new EventEmitter();
    const room = Object.assign(new EventEmitter(), {
        peer: { uuid: BACKEND, setProperty: (key: string, value: string) => { properties.set(key, value); } },
        peers: new Map([[SPEAKER, { uuid: SPEAKER }]]),
        room: { uuid: joined ? ROOM : '' },
    });
    const socket = { writableLength: 0, destroyed: false, writable: true };
    const writes: { id: number; payload: Record<string, unknown> }[] = [];
    const registered = new Set<unknown>();
    const scene = {
        connections: [{ socket }],
        register: (component: unknown) => { registered.add(component); },
        unregister: (component: unknown) => { registered.delete(component); },
        send: (id: { b: number }, payload: Record<string, unknown>) => { writes.push({ id: id.b, payload }); },
    };
    const logs: string[] = [];
    const controller = new MotionReplayController(scene as unknown as NetworkScene, room, service, 16000,
        (message) => logs.push(message));
    t.after(() => controller.close());
    const ping = (overrides: Record<string, unknown> = {}) => controller.processMessage({
        message: Buffer.from(JSON.stringify({
            version: 1, type: 'ping', backendUuid: BACKEND, sessionId: controller.sessionId,
            peerUuid: SPEAKER, nonce: 'nonce-1', t0: 123.5, ...overrides,
        })),
    });
    return { properties, service, room, socket, writes, registered, controller, logs, ping };
}

test('motion capability is published only after joining and removed on service close', (t) => {
    const h = harness(t, false);
    assert.equal(h.registered.size, 1);
    assert.equal(h.properties.size, 0);
    h.room.room.uuid = ROOM;
    h.room.emit('OnJoinedRoom');
    assert.deepEqual(JSON.parse(h.properties.get(MOTION_REPLAY_PROPERTY)!), {
        version: 1, sessionId: h.controller.sessionId,
    });
    h.service.emit('translationClosed');
    assert.equal(h.properties.get(MOTION_REPLAY_PROPERTY), '');
    assert.equal(h.registered.size, 0);
    assert.equal(h.service.listenerCount('motionReplay'), 0);
    assert.equal(h.room.listenerCount('OnJoinedRoom'), 0);
    h.controller.close();
});

test('lifecycle sideband preserves utterance identity, sample counts and order on ID 101', (t) => {
    const h = harness(t);
    h.service.emit('motionReplay', accepted);
    h.service.emit('motionReplay', play);
    h.service.emit('motionReplay', {
        type: 'terminal', requestId: 'request-1', speakerUuid: SPEAKER, status: 'sent',
    });
    assert.deepEqual(h.writes.map((write) => write.payload.type), ['accepted', 'play', 'terminal']);
    for (const write of h.writes) {
        assert.equal(write.id, MOTION_REPLAY_NETWORK_ID);
        assert.equal(write.payload.version, 1);
        assert.equal(write.payload.backendUuid, BACKEND);
        assert.equal(write.payload.sessionId, h.controller.sessionId);
        assert.equal(write.payload.requestId, 'request-1');
        assert.equal(write.payload.speakerUuid, SPEAKER);
    }
    assert.equal(h.writes[1].payload.sampleCount, 16000);
    assert.equal(h.writes[1].payload.audioLength, 32000);
});

test('play metadata reserves capacity for its following legacy AudioInfo header', (t) => {
    const h = harness(t);
    const payload = { version: 1, backendUuid: BACKEND, sessionId: h.controller.sessionId, ...play };
    const bytes = Buffer.byteLength(JSON.stringify(payload));
    // Metadata by itself fits, but the header following it would exceed the cap.
    h.socket.writableLength = 32000 - bytes - 12;
    assert.throws(() => h.service.emit('motionReplay', play), /congested/);
    assert.equal(h.writes.length, 0);
    h.socket.writableLength = 0;
    h.service.emit('motionReplay', play);
    assert.equal(h.writes.length, 1);
});

test('unavailable accepted metadata is discarded and required play metadata fails closed', (t) => {
    const h = harness(t);
    h.socket.destroyed = true;
    assert.doesNotThrow(() => h.service.emit('motionReplay', accepted));
    assert.doesNotThrow(() => h.service.emit('motionReplay', {
        type: 'terminal', requestId: 'request-1', speakerUuid: SPEAKER, status: 'cancelled',
    }));
    assert.throws(() => h.service.emit('motionReplay', play), /writable/);
    assert.equal(h.writes.length, 0);
    assert.equal(h.logs.length, 2);
});

test('terminal metadata retries after congestion clears so pinned clips can be released', async (t) => {
    const h = harness(t);
    h.socket.writableLength = 32000;
    h.service.emit('motionReplay', { type: 'terminal', requestId: 'request-1', speakerUuid: SPEAKER, status: 'failed' });
    assert.equal(h.writes.length, 0);
    h.socket.writableLength = 0;
    await delay(300);
    assert.equal(h.writes.length, 1);
    assert.equal(h.writes[0].payload.type, 'terminal');
    assert.equal(h.writes[0].payload.requestId, 'request-1');
    await delay(260);
    assert.equal(h.writes.length, 1, 'a successful retry is removed');
});

test('terminal retry capacity is bounded and overflow changes the advertised session to clear client state', (t) => {
    const h = harness(t);
    const oldSession = h.controller.sessionId;
    h.socket.writableLength = 32000;
    for (let index = 0; index < 33; index++) {
        h.service.emit('motionReplay', {
            type: 'terminal', requestId: `failed-${index}`, speakerUuid: SPEAKER, status: 'failed',
        });
    }
    assert.notEqual(h.controller.sessionId, oldSession);
    assert.equal(JSON.parse(h.properties.get(MOTION_REPLAY_PROPERTY)!).sessionId, h.controller.sessionId);
    h.socket.writableLength = 0;
    h.service.emit('motionReplay', accepted);
    assert.equal(h.writes.length, 1, 'old-session terminal retries were invalidated');
});

test('ping requires the selected backend/session and a current room peer', (t) => {
    const h = harness(t);
    for (const invalid of [
        { backendUuid: SPEAKER }, { sessionId: 'stale-session' }, { peerUuid: BACKEND },
        { version: 2 }, { nonce: '' }, { nonce: 'x'.repeat(129) }, { t0: -1 }, { t0: null }, { type: 'pong' },
    ]) h.ping(invalid);
    h.controller.processMessage({ message: Buffer.from('invalid JSON') });
    h.controller.processMessage({ message: Buffer.alloc(2049) });
    assert.equal(h.writes.length, 0);
    h.ping();
    assert.equal(h.writes.length, 1);
    const pong = h.writes[0].payload;
    assert.equal(pong.type, 'pong');
    assert.equal(pong.peerUuid, SPEAKER);
    assert.equal(pong.nonce, 'nonce-1');
    assert.equal(pong.t0, 123.5);
    assert.ok(typeof pong.s1 === 'number' && Number.isFinite(pong.s1));
    assert.ok(typeof pong.s2 === 'number' && pong.s2 >= (pong.s1 as number));
    h.ping({ nonce: 'too-soon' });
    assert.equal(h.writes.length, 1, 'a burst from one peer is bounded');
});

test('room transitions discard ping rate state and stale sessions cannot calibrate another backend', (t) => {
    const first = harness(t);
    const second = harness(t);
    assert.notEqual(first.controller.sessionId, second.controller.sessionId);
    first.ping({ sessionId: second.controller.sessionId });
    assert.equal(first.writes.length, 0);
    first.ping();
    first.room.emit('OnJoinedRoom');
    first.ping({ nonce: 'after-join' });
    assert.equal(first.writes.length, 2);
    first.room.room.uuid = '';
    first.room.emit('OnJoinedRoom');
    first.ping({ nonce: 'after-leave' });
    assert.equal(first.writes.length, 2);
});
