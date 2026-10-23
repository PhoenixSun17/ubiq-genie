using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using Ubiq.Avatars;
using Ubiq.Messaging;
using Ubiq.Rooms;
using Ubiq.Samples;
using Ubiq.Voip;
using UnityEditor;
using UnityEngine;
using Avatar = Ubiq.Avatars.Avatar;
using Object = UnityEngine.Object;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion.Tests
{
    public sealed class RemoteAvatarReplayTests
    {
        private readonly List<GameObject> objects = new List<GameObject>();

        private sealed class Peer : IPeer
        {
            public string uuid { get; set; }
            public NetworkId networkId => default;
            public string this[string key] { get => null; set { } }
            public IEnumerator<KeyValuePair<string, string>> GetEnumerator() =>
                ((IEnumerable<KeyValuePair<string, string>>)Array.Empty<KeyValuePair<string, string>>()).GetEnumerator();
            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        }

        [TearDown]
        public void TearDown()
        {
            for (var i = objects.Count - 1; i >= 0; --i)
                if (objects[i]) Object.DestroyImmediate(objects[i]);
            objects.Clear();
        }

        [Test]
        public void ThirtySecondHistoryHasFixedCapacityAndRejectsExpiredIntervals()
        {
            var history = new MotionPoseBuffer(30, 60, 1, 1);
            var frame = Frame(0);
            for (var i = 0; i <= 6000; ++i)
            {
                frame.Positions[0] = new Vector3(i, 0, 0);
                history.Add(i / 60.0, frame);
            }
            Assert.That(history.Capacity, Is.EqualTo(1802));
            Assert.That(history.Count, Is.LessThanOrEqualTo(history.Capacity));
            Assert.That(history.TryPin(69.9, 80, out _), Is.False);
            Assert.That(history.TryPin(70, 100, out var clip), Is.True);
            Assert.That(clip.StartTime, Is.EqualTo(70));
            Assert.That(clip.EndTime, Is.EqualTo(100));
        }

        [Test]
        public void PinnedSentenceRemainsIntactAfterRingWrapAndClear()
        {
            var history = new MotionPoseBuffer(3, 2, 1, 1);
            for (var i = 0; i <= 6; ++i) history.Add(i / 2.0, Frame(i));
            Assert.That(history.TryPin(0.25, 2.75, out var clip), Is.True);
            for (var i = 7; i <= 40; ++i) history.Add(i / 2.0, Frame(i));
            history.Clear();
            var result = Frame(0);
            clip.Sample(0.5, result);
            Assert.That(result.Positions[0].x, Is.EqualTo(3).Within(0.0001));
            clip.Sample(0, result);
            Assert.That(result.Positions[0].x, Is.EqualTo(0.5).Within(0.0001));
            clip.Sample(1, result);
            Assert.That(result.Positions[0].x, Is.EqualTo(5.5).Within(0.0001));
        }

        [TestCase(-0.1, 1)]
        [TestCase(0, 2.1)]
        [TestCase(1, 1)]
        [TestCase(1, 0)]
        [TestCase(double.NaN, 1)]
        [TestCase(0, double.PositiveInfinity)]
        public void PinRequiresFiniteCompleteForwardInterval(double start, double end)
        {
            var history = new MotionPoseBuffer(30, 60, 1, 1);
            history.Add(0, Frame(0));
            history.Add(2, Frame(2));
            Assert.That(history.TryPin(start, end, out _), Is.False);
        }

        [Test]
        public void SampleUsesWholeSentenceProgressAndPreservesStationaryUnderflow()
        {
            var history = new MotionPoseBuffer(30, 60, 2, 1);
            var first = new MotionPoseFrame(2, 1);
            first.Visible[0] = true;
            var last = new MotionPoseFrame(2, 1);
            last.Visible[0] = true;
            last.Positions[0] = new Vector3(8, 2, -4);
            last.Rotations[0] = Quaternion.Euler(0, 120, 0);
            last.Rotations[1] = Quaternion.Euler(80, 0, 0);
            history.Add(0, first);
            history.Add(4, last);
            Assert.That(history.TryPin(0, 4, out var clip), Is.True);
            var result = new MotionPoseFrame(2, 1);
            clip.Sample(0.25, result);
            Assert.That(result.Positions[0], Is.EqualTo(new Vector3(2, 0.5f, -1)));
            Assert.That(Quaternion.Angle(result.Rotations[0], Quaternion.Euler(0, 30, 0)), Is.LessThan(0.01));
            Assert.That(Quaternion.Angle(result.Rotations[1], Quaternion.Euler(20, 0, 0)), Is.LessThan(0.01));
            var held = result.Positions[0];
            clip.Sample(0.25, result);
            Assert.That(result.Positions[0], Is.EqualTo(held));
            clip.Sample(10, result);
            Assert.That(result.Positions[0], Is.EqualTo(last.Positions[0]));
        }

        [Test]
        public void RingCopiesFramesAndRejectsBackwardsOrDuplicateTime()
        {
            var history = new MotionPoseBuffer(30, 60, 1, 1);
            var frame = Frame(1);
            history.Add(1, frame);
            frame.Positions[0] = new Vector3(3, 0, 0);
            history.Add(2, frame);
            frame.Positions[0] = new Vector3(100, 0, 0);
            Assert.Throws<ArgumentException>(() => history.Add(2, frame));
            Assert.Throws<ArgumentException>(() => history.Add(0, frame));
            Assert.That(history.TryPin(1, 2, out var clip), Is.True);
            var result = Frame(0);
            clip.Sample(0, result);
            Assert.That(result.Positions[0].x, Is.EqualTo(1));
            clip.Sample(1, result);
            Assert.That(result.Positions[0].x, Is.EqualTo(3));
        }

        [TestCase(true)]
        [TestCase(false)]
        public void LocalAvatarFlagOrLocalUuidPreventsProxyAndAnimationChanges(bool localFlag)
        {
            var owner = NewObject("replay");
            var replay = owner.AddComponent<RemoteAvatarReplay>();
            var room = TestRoom();
            SetField(replay, "room", room);
            SetField(replay, "roomRoot", room.transform);
            var avatarObject = NewObject("local");
            var avatar = avatarObject.AddComponent<Avatar>();
            avatar.IsLocal = localFlag;
            avatar.SetPeer(new Peer { uuid = localFlag ? "different-peer" : room.Me.uuid });
            var animator = avatarObject.AddComponent<Animator>();
            animator.cullingMode = AnimatorCullingMode.CullCompletely;
            var position = avatarObject.transform.position = new Vector3(1, 2, 3);
            Invoke(replay, "OnAvatarCreated", avatar);
            replay.Present("sentence", avatar.Peer.uuid, 0.5);
            Invoke(replay, "LateUpdate");
            Assert.That(room.transform.childCount, Is.Zero);
            Assert.That(avatarObject.transform.position, Is.EqualTo(position));
            Assert.That(animator.cullingMode, Is.EqualTo(AnimatorCullingMode.CullCompletely));
            Assert.That(replay.Pin("sentence", avatar.Peer.uuid, 0, 1, out _), Is.False);
        }

        [Test]
        public void RemoteProxyContainsOnlyRenderingComponentsAndTeardownRestoresCulling()
        {
            var room = TestRoom();
            var replay = NewObject("replay").AddComponent<RemoteAvatarReplay>();
            SetField(replay, "room", room);
            SetField(replay, "roomRoot", room.transform);
            var avatarObject = NewObject("remote");
            var avatar = avatarObject.AddComponent<Avatar>();
            avatar.SetPeer(new Peer { uuid = "remote-peer" });
            var model = NewObject("model");
            model.SetActive(false);
            model.transform.SetParent(avatarObject.transform);
            var floating = model.AddComponent<FloatingAvatar>();
            floating.head = Visual(model.transform, "head", out floating.headRenderer);
            floating.torso = Visual(model.transform, "torso", out floating.torsoRenderer);
            var hand = NewObject("left");
            hand.transform.SetParent(model.transform);
            var bone = NewObject("finger");
            bone.transform.SetParent(hand.transform);
            var skin = hand.AddComponent<SkinnedMeshRenderer>();
            skin.bones = new[] { bone.transform };
            skin.rootBone = bone.transform;
            skin.updateWhenOffscreen = false;
            floating.leftHand = hand.transform;
            floating.leftHandRenderer = skin;
            floating.rightHand = Visual(model.transform, "right", out floating.rightHandRenderer);
            var animator = model.AddComponent<Animator>();
            animator.cullingMode = AnimatorCullingMode.CullCompletely;
            Invoke(replay, "OnAvatarCreated", avatar);
            Assert.That(room.transform.childCount, Is.EqualTo(1));
            var proxy = room.transform.GetChild(0);
            Assert.That(proxy.GetComponentsInChildren<MonoBehaviour>(true), Is.Empty);
            Assert.That(proxy.GetComponentsInChildren<Animator>(true), Is.Empty);
            Assert.That(proxy.GetComponentsInChildren<Collider>(true), Is.Empty);
            Assert.That(proxy.GetComponentsInChildren<Renderer>(true).Length, Is.EqualTo(4));
            var proxySkin = proxy.GetComponentInChildren<SkinnedMeshRenderer>(true);
            Assert.That(proxySkin.bones[0], Is.Not.SameAs(bone.transform));
            Assert.That(proxySkin.bones[0].IsChildOf(proxy), Is.True);
            Assert.That(proxySkin.rootBone, Is.SameAs(proxySkin.bones[0]));
            Assert.That(skin.updateWhenOffscreen, Is.True);
            Assert.That(animator.cullingMode, Is.EqualTo(AnimatorCullingMode.AlwaysAnimate));
            replay.Initialize(null, null);
            Assert.That(room.transform.childCount, Is.Zero);
            Assert.That(skin.updateWhenOffscreen, Is.False);
            Assert.That(animator.cullingMode, Is.EqualTo(AnimatorCullingMode.CullCompletely));
        }

        [Test]
        public void OriginalVoiceToggleChangesReceiverOnlyAndRestoresPriorMute()
        {
            var toggle = NewObject("toggle").AddComponent<OriginalVoiceToggle>();
            var capture = NewObject("microphone").AddComponent<AudioSource>();
            var connection = NewObject("receiver").AddComponent<VoipPeerConnection>();
            var receive = connection.gameObject.AddComponent<AudioSource>();
            toggle.playOriginalVoice = false;
            Invoke(toggle, "OnPeerConnection", connection);
            Assert.That(receive.mute, Is.True);
            Assert.That(capture.mute, Is.False);
            toggle.Initialize(null);
            Assert.That(receive.mute, Is.False);
            Assert.That(capture.mute, Is.False);
        }

        [Test]
        public void ActualFloatingPrefabCopiesItsMeshesAndBonesWithoutNetworkComponents()
        {
            var replay = ActualAvatar(out var room, out var avatar, out var floating, out var state);
            var proxy = GetField<GameObject>(state, "Proxy");
            var sources = GetField<Transform[]>(state, "Sources");
            var targets = GetField<Transform[]>(state, "Targets");
            var sourceRenderers = GetField<Renderer[]>(state, "BodyRenderers");
            var proxyRenderers = GetField<Renderer[]>(state, "ProxyRenderers");
            Assert.That(proxy.transform.parent, Is.EqualTo(room.transform));
            Assert.That(proxy.GetComponentsInChildren<MonoBehaviour>(true), Is.Empty);
            Assert.That(proxy.GetComponentsInChildren<Animator>(true), Is.Empty);
            Assert.That(proxy.GetComponentsInChildren<Collider>(true), Is.Empty);
            var boneCount = 0;
            for (var i = 0; i < sourceRenderers.Length; ++i)
            {
                CollectionAssert.AreEqual(sourceRenderers[i].sharedMaterials, proxyRenderers[i].sharedMaterials);
                if (sourceRenderers[i] is SkinnedMeshRenderer sourceSkin)
                {
                    var proxySkin = (SkinnedMeshRenderer)proxyRenderers[i];
                    Assert.That(proxySkin.sharedMesh, Is.SameAs(sourceSkin.sharedMesh));
                    Assert.That(proxySkin.bones.Length, Is.EqualTo(sourceSkin.bones.Length).And.GreaterThan(0));
                    boneCount += sourceSkin.bones.Length;
                    for (var bone = 0; bone < sourceSkin.bones.Length; ++bone)
                    {
                        var sourceIndex = Array.IndexOf(sources, sourceSkin.bones[bone]);
                        Assert.That(sourceIndex, Is.GreaterThanOrEqualTo(0));
                        Assert.That(proxySkin.bones[bone], Is.SameAs(targets[sourceIndex]));
                        Assert.That(proxySkin.bones[bone].IsChildOf(proxy.transform), Is.True);
                        Assert.That(proxySkin.bones[bone].IsChildOf(avatar.transform), Is.False);
                    }
                    if (sourceSkin.rootBone)
                        Assert.That(proxySkin.rootBone, Is.SameAs(targets[Array.IndexOf(sources, sourceSkin.rootBone)]));
                    else
                        Assert.That(proxySkin.rootBone, Is.Null);
                }
                else
                {
                    Assert.That(proxyRenderers[i].GetComponent<MeshFilter>().sharedMesh,
                        Is.SameAs(sourceRenderers[i].GetComponent<MeshFilter>().sharedMesh));
                }
            }
            Assert.That(floating.leftHandRenderer, Is.InstanceOf<SkinnedMeshRenderer>());
            Assert.That(floating.rightHandRenderer, Is.InstanceOf<SkinnedMeshRenderer>());
            Invoke(replay, "CaptureLive", state, 1.0);
            var live = GetField<MotionPoseFrame>(state, "Live");
            Assert.That(live.Positions.Length, Is.EqualTo(sources.Length));
            var history = GetField<MotionPoseBuffer>(state, "History");
            var payloadBytes = (long)history.Capacity * (sources.Length * 40L + 4 + 8);
            TestContext.WriteLine($"Actual Ubiq Floating Avatar: {sources.Length} captured transforms, " +
                $"{boneCount} skinned-hand bone references, {history.Capacity} ring frames; " +
                $"{payloadBytes} bytes of pose-array/time payload per avatar, excluding managed-object overhead and pinned clips.");
        }

        [Test]
        public void TenPinnedSentencesAreBoundedAndFinishCancelResetReleaseCapacity()
        {
            var replay = ActualAvatar(out _, out _, out var floating, out var state);
            FillHistory(replay, floating, state, out var start, out var end);
            for (var i = 0; i < 10; ++i)
                Assert.That(replay.Pin("sentence-" + i, "remote-peer", start, end, out var reason), Is.True, reason);
            Assert.That(replay.Pin("overflow", "remote-peer", start, end, out var fullReason), Is.False);
            StringAssert.Contains("full", fullReason);
            // Redelivery is idempotent even when all ten slots are occupied.
            Assert.That(replay.Pin("sentence-0", "remote-peer", start, end, out _), Is.True);
            replay.Finish("sentence-0");
            Assert.That(replay.Pin("replacement", "remote-peer", start, end, out _), Is.True);
            Assert.That(replay.Pin("overflow", "remote-peer", start, end, out _), Is.False);
            replay.CancelSpeaker("remote-peer");
            Assert.That(GetField<IDictionary>(replay, "pinned").Count, Is.Zero);
            Assert.That(GetField<MotionPoseBuffer>(state, "History").Count, Is.Zero);
            Assert.That(replay.Pin("expired", "remote-peer", start, end, out _), Is.False);
            FillHistory(replay, floating, state, out start, out end);
            Assert.That(replay.Pin("after-cancel", "remote-peer", start, end, out _), Is.True);
            replay.ResetReplay();
            Assert.That(GetField<IDictionary>(replay, "pinned").Count, Is.Zero);
            Assert.That(GetField<MotionPoseBuffer>(state, "History").Count, Is.Zero);
        }

        [Test]
        public void PresentReplaysRoomSpacePoseAndBonesThenFinishBlendsBackWithoutWritingLiveTransforms()
        {
            var replay = ActualAvatar(out var room, out _, out var floating, out var state);
            room.transform.SetPositionAndRotation(new Vector3(100, 0, -50), Quaternion.Euler(0, 90, 0));
            FillHistory(replay, floating, state, out var start, out var end);
            var sources = GetField<Transform[]>(state, "Sources");
            var targets = GetField<Transform[]>(state, "Targets");
            var sourceBone = ((SkinnedMeshRenderer)floating.leftHandRenderer).bones[0];
            var proxyBone = targets[Array.IndexOf(sources, sourceBone)];
            var liveHead = floating.head.position;
            var liveBone = sourceBone.localRotation;
            var renderers = GetField<Renderer[]>(state, "HiddenRenderers");
            // Check restoration preserves a renderer that was already hidden before playback.
            var lastRenderer = renderers[renderers.Length - 1];
            lastRenderer.forceRenderingOff = true;
            var priorFlags = new bool[renderers.Length];
            for (var i = 0; i < renderers.Length; ++i) priorFlags[i] = renderers[i].forceRenderingOff;
            Assert.That(replay.Pin("sentence", "remote-peer", start, end, out var reason), Is.True, reason);
            SetField(replay, "subscribed", true);
            SetField(state, "NextCapture", double.PositiveInfinity);
            replay.Present("sentence", "remote-peer", 0.5);
            Invoke(replay, "LateUpdate");
            Assert.That(GetField<GameObject>(state, "Proxy").activeSelf, Is.True);
            Assert.That(replay.TryGetHead("remote-peer", out var firstBlendPose), Is.True);
            Assert.That(Vector3.Distance(firstBlendPose.position, liveHead), Is.LessThan(0.0001),
                "The first blend frame must start at the current live position.");
            foreach (var renderer in renderers) Assert.That(renderer.forceRenderingOff, Is.True);
            SetField(state, "BlendStarted", Time.realtimeSinceStartupAsDouble - 1.0);
            Invoke(replay, "LateUpdate");
            Assert.That(replay.TryGetHead("remote-peer", out var replayPose), Is.True);
            var expected = room.transform.TransformPoint(new Vector3(2, 1, 0));
            Assert.That(Vector3.Distance(replayPose.position, expected), Is.LessThan(0.0001));
            Assert.That(Quaternion.Angle(proxyBone.localRotation, Quaternion.Euler(0, 30, 0)), Is.LessThan(0.01));
            Assert.That(floating.head.position, Is.EqualTo(liveHead));
            Assert.That(Quaternion.Angle(sourceBone.localRotation, liveBone), Is.LessThan(0.01));
            // The same consumed-audio position must hold motion during an underrun.
            replay.Present("sentence", "remote-peer", 0.5);
            Invoke(replay, "LateUpdate");
            Assert.That(replay.TryGetHead("remote-peer", out var held), Is.True);
            Assert.That(held.position, Is.EqualTo(replayPose.position));
            replay.Finish("sentence");
            Assert.That(GetField<IDictionary>(replay, "pinned").Count, Is.Zero);
            Assert.That(GetField<bool>(state, "Releasing"), Is.True);
            SetField(state, "BlendStarted", Time.realtimeSinceStartupAsDouble - 1.0);
            Invoke(replay, "LateUpdate");
            Assert.That(GetField<GameObject>(state, "Proxy").activeSelf, Is.False);
            Assert.That(replay.TryGetHead("remote-peer", out var restored), Is.True);
            Assert.That(restored.position, Is.EqualTo(liveHead));
            for (var i = 0; i < renderers.Length; ++i)
                Assert.That(renderers[i].forceRenderingOff, Is.EqualTo(priorFlags[i]));
        }

        private static MotionPoseFrame Frame(float position)
        {
            var frame = new MotionPoseFrame(1, 1);
            frame.Positions[0] = new Vector3(position, 0, 0);
            frame.Visible[0] = true;
            return frame;
        }

        private GameObject NewObject(string name)
        {
            var result = new GameObject(name);
            objects.Add(result);
            return result;
        }

        private RoomClient TestRoom()
        {
            // Avoid NetworkScene's root singleton/DontDestroyOnLoad path in an EditMode fixture.
            var room = NewObject("room");
            room.transform.SetParent(NewObject("fixture").transform);
            return room.AddComponent<RoomClient>();
        }

        private RemoteAvatarReplay ActualAvatar(out RoomClient room, out Avatar avatar,
            out FloatingAvatar floating, out object state)
        {
            const string path = "Packages/com.ucl.ubiq/Runtime/ExampleAvatars/Floating/Ubiq Floating Avatar.prefab";
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            Assert.That(prefab, Is.Not.Null, "The project's configured Ubiq floating avatar must be available.");
            var inactiveParent = NewObject("inactive prefab fixture");
            inactiveParent.SetActive(false);
            var instance = Object.Instantiate(prefab, inactiveParent.transform, false);
            objects.Add(instance);
            avatar = instance.GetComponent<Avatar>();
            Assert.That(avatar, Is.Not.Null);
            avatar.IsLocal = false;
            avatar.SetPeer(new Peer { uuid = "remote-peer" });
            floating = instance.GetComponentInChildren<FloatingAvatar>(true);
            Assert.That(floating, Is.Not.Null);
            room = TestRoom();
            var replay = NewObject("replay").AddComponent<RemoteAvatarReplay>();
            SetField(replay, "room", room);
            SetField(replay, "roomRoot", room.transform);
            Invoke(replay, "OnAvatarCreated", avatar);
            var states = GetField<IDictionary>(replay, "avatars");
            Assert.That(states.Contains("remote-peer"), Is.True,
                "Proxy construction must support the actual configured prefab, not only a synthetic model.");
            state = states["remote-peer"];
            return replay;
        }

        private static void FillHistory(RemoteAvatarReplay replay, FloatingAvatar floating, object state,
            out double start, out double end)
        {
            var roomRoot = GetField<Transform>(replay, "roomRoot");
            var history = GetField<MotionPoseBuffer>(state, "History");
            history.Clear();
            start = Time.realtimeSinceStartupAsDouble - 3;
            end = start + 2;
            var bone = ((SkinnedMeshRenderer)floating.leftHandRenderer).bones[0];
            floating.head.position = roomRoot.TransformPoint(new Vector3(0, 1, 0));
            bone.localRotation = Quaternion.identity;
            Invoke(replay, "CaptureLive", state, start);
            history.Add(start, GetField<MotionPoseFrame>(state, "Live"));
            floating.head.position = roomRoot.TransformPoint(new Vector3(4, 1, 0));
            bone.localRotation = Quaternion.Euler(0, 60, 0);
            Invoke(replay, "CaptureLive", state, end);
            history.Add(end, GetField<MotionPoseFrame>(state, "Live"));
            floating.head.position = roomRoot.TransformPoint(new Vector3(10, 2, 0));
            bone.localRotation = Quaternion.Euler(0, 120, 0);
        }

        private Transform Visual(Transform parent, string name, out Renderer renderer)
        {
            var visual = NewObject(name);
            visual.transform.SetParent(parent);
            visual.AddComponent<MeshFilter>();
            renderer = visual.AddComponent<MeshRenderer>();
            return visual.transform;
        }

        private static void Invoke(object target, string method, params object[] args) =>
            target.GetType().GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance).Invoke(target, args);

        private static void SetField(object target, string name, object value) =>
            target.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance).SetValue(target, value);

        private static T GetField<T>(object target, string name) =>
            (T)target.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance).GetValue(target);
    }
}
