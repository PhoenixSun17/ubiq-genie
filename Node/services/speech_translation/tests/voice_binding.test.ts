import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test, type TestContext } from 'node:test';
import {
    VoiceBindingController, VOICE_CATALOG_PROPERTY, VOICE_SELECTION_PROPERTY, VOICE_BINDINGS_PROPERTY,
} from '../voice_binding';
import type { VoiceCatalogEntry } from '../types';

const NODE = '11111111-1111-4111-8111-111111111111';
const A = '22222222-2222-4222-8222-222222222222';
const B = '33333333-3333-4333-8333-333333333333';
const ROOM = '44444444-4444-4444-8444-444444444444';
const REQUEST = '55555555-5555-4555-8555-555555555555';
const NEXT_REQUEST = '66666666-6666-4666-8666-666666666666';
const ID = 'a'.repeat(64);
const OTHER_ID = 'b'.repeat(64);

function harness(t: TestContext, joined = true) {
    const published = new Map<string, string>();
    const writes: string[] = [];
    const peer = {
        uuid: NODE, properties: new Map<string, string>(),
        setProperty: (key: string, value: string) => { published.set(key, value); writes.push(key); },
    };
    const room = Object.assign(new EventEmitter(), {
        peer, room: { uuid: joined ? ROOM : '' },
        peers: new Map([[NODE, peer]]),
    });
    const profiles: VoiceCatalogEntry[] = [{ id: ID, name: 'my-voice.wav', language: 'en' }];
    const effective = new Map<string, string>();
    const calls: string[] = [];
    const service = Object.assign(new EventEmitter(), {
        listVoices: () => profiles.map((entry) => ({ ...entry })),
        getBoundVoiceId: (uuid: string) => effective.get(uuid),
        bindVoice: (uuid: string, id: string) => {
            if (!room.peers.has(uuid) || !profiles.some((entry) => entry.id === id)) throw new Error('Unavailable');
            calls.push(uuid);
            effective.set(uuid, id);
        },
    });
    const logs: string[] = [];
    const controller = new VoiceBindingController(room, service, (message) => logs.push(message));
    t.after(() => controller.close());
    const add = (uuid: string, raw = '') => {
        const next = { uuid, properties: new Map([[VOICE_SELECTION_PROPERTY, raw]]), setProperty: () => {} };
        room.peers.set(uuid, next);
        room.emit('OnPeerAdded', next);
        return next;
    };
    const select = (uuid: string, values: Record<string, unknown> = {}) => {
        const next = room.peers.get(uuid)!;
        next.properties.set(VOICE_SELECTION_PROPERTY, JSON.stringify({
            version: 1, requestId: REQUEST, voiceId: ID, backendUuid: NODE, ...values,
        }));
        room.emit('OnPeerUpdated', next);
    };
    const statuses = () => JSON.parse(published.get(VOICE_BINDINGS_PROPERTY)!).bindings as {
        peerUuid: string; voiceId: string; requestId: string; status: string; message: string;
    }[];
    return { room, profiles, service, effective, calls, logs, controller, published, writes, add, select, statuses };
}

test('publishes only public catalog metadata after room admission; replays existing peer choice', (t) => {
    const h = harness(t, false);
    h.add(A);
    h.select(A);
    assert.equal(h.calls.length, 0);
    assert.equal(h.published.size, 0);
    Object.assign(h.profiles[0], { refAudioPath: 'private/path.wav', promptText: 'Private transcript' });
    h.room.room.uuid = ROOM;
    h.room.emit('OnJoinedRoom', h.room.room);
    assert.deepEqual(JSON.parse(h.published.get(VOICE_CATALOG_PROPERTY)!), {
        version: 1, voices: [{ id: ID, name: 'my-voice.wav', language: 'en' }],
    });
    assert.deepEqual(h.calls, [A]);
    assert.equal(h.statuses()[0].status, 'bound');
});

test('binds event peer UUID, never a payload UUID, and ignores unrelated property updates', (t) => {
    const h = harness(t);
    const a = h.add(A);
    h.add(B);
    h.select(A, { peerUuid: B });
    assert.equal(h.effective.get(A), ID);
    assert.equal(h.effective.has(B), false);
    const writes = h.writes.length;
    a.properties.set('ubiq.displayname', 'Changed name');
    h.room.emit('OnPeerUpdated', a);
    assert.deepEqual(h.calls, [A]);
    assert.equal(h.writes.length, writes);
    assert.deepEqual(h.statuses().map((entry) => entry.peerUuid), [A]);
});

test('selection is targeted to one backend; new peer UUID can bind the same remembered voice', (t) => {
    const h = harness(t);
    h.add(A);
    h.select(A, { backendUuid: B });
    assert.equal(h.calls.length, 0);
    h.select(A);
    h.room.peers.delete(A);
    h.room.emit('OnPeerRemoved', { uuid: A });
    assert.deepEqual(h.statuses(), []);
    h.add(B);
    h.select(B, { requestId: NEXT_REQUEST });
    assert.deepEqual(h.calls, [A, B]);
    assert.equal(h.statuses()[0].requestId, NEXT_REQUEST);
    assert.equal(h.statuses()[0].peerUuid, B);
});

test('invalid, oversized and unsupported requests do not alter a valid binding', (t) => {
    const h = harness(t);
    const a = h.add(A);
    h.select(A);
    for (const raw of ['{', '{}', 'x'.repeat(1025), 'null', '[]']) {
        a.properties.set(VOICE_SELECTION_PROPERTY, raw);
        h.room.emit('OnPeerUpdated', a);
    }
    h.select(A, { version: 2 });
    h.select(A, { voiceId: 'not-an-id' });
    h.select(A, { requestId: 'not-a-request' });
    assert.deepEqual(h.calls, [A]);
    assert.equal(h.effective.get(A), ID);
    h.select(A, { voiceId: OTHER_ID, requestId: NEXT_REQUEST });
    assert.equal(h.statuses()[0].status, 'error');
    assert.equal(h.statuses()[0].requestId, NEXT_REQUEST);
    assert.equal(h.effective.get(A), ID);
});

test('binding failure exposes no private backend error and never reports a false success', (t) => {
    const h = harness(t);
    h.add(A);
    h.effective.set(A, ID); // A legacy mapping alone is not an accepted selection request.
    h.service.bindVoice = () => { throw new Error('Private path C:/secret/reference.wav'); };
    h.select(A);
    h.service.emit('voiceCatalogChanged');
    assert.equal(h.statuses()[0].status, 'error');
    assert.equal(h.published.get(VOICE_BINDINGS_PROPERTY)!.includes('secret'), false);
});

test('catalog deletion revokes confirmation; restoration can confirm the accepted selection again', (t) => {
    const h = harness(t);
    h.add(A);
    h.select(A);
    h.profiles.length = 0;
    h.effective.delete(A);
    h.service.emit('voiceCatalogChanged');
    assert.equal(h.statuses()[0].status, 'error');
    assert.deepEqual(JSON.parse(h.published.get(VOICE_CATALOG_PROPERTY)!).voices, []);
    h.profiles.push({ id: ID, name: 'my-voice.wav', language: 'en' });
    h.effective.set(A, ID);
    h.service.emit('voiceCatalogChanged');
    assert.equal(h.statuses()[0].status, 'bound');
});

test('backend restart requires a request addressed to its new UUID', (t) => {
    const h = harness(t);
    h.add(A);
    h.select(A);
    h.controller.close();
    h.room.peer.uuid = B;
    h.room.peers.delete(NODE);
    h.room.peers.set(B, h.room.peer);
    h.effective.clear();
    const restarted = new VoiceBindingController(h.room, h.service, (message) => h.logs.push(message));
    t.after(() => restarted.close());
    assert.equal(h.effective.has(A), false, 'old backend address must not be reused');
    h.select(A, { backendUuid: B, requestId: NEXT_REQUEST });
    assert.equal(h.effective.get(A), ID);
    assert.equal(h.statuses()[0].requestId, NEXT_REQUEST);
});

test('leaving and closing remove status and subscriptions without changing files', (t) => {
    const h = harness(t);
    h.add(A);
    h.select(A);
    h.room.peers.delete(A);
    h.room.emit('OnPeerRemoved', { uuid: A });
    assert.deepEqual(h.statuses(), []);
    h.controller.close();
    const writes = h.writes.length;
    h.service.emit('voiceCatalogChanged');
    h.room.emit('OnJoinedRoom');
    h.add(A);
    h.select(A);
    assert.equal(h.writes.length, writes);
    assert.equal(h.room.listenerCount('OnPeerUpdated'), 0);
    assert.equal(h.service.listenerCount('voiceCatalogChanged'), 0);
});
