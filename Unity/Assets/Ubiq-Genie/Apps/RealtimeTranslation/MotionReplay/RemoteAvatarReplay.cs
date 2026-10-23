using System;
using System.Collections.Generic;
using Ubiq.Avatars;
using Ubiq.Messaging;
using Ubiq.Rooms;
using Ubiq.Samples;
using UnityEngine;
using Avatar = Ubiq.Avatars.Avatar;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Records live remote visuals and presents sentence motion without writing to any tracked avatar.</summary>
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(10000)]
    public sealed class RemoteAvatarReplay : MonoBehaviour
    {
        [SerializeField, Min(1)] private float historySeconds = 30;
        [SerializeField, Range(1, 120)] private int captureRate = 60;
        [SerializeField, Min(0)] private float blendSeconds = 0.2f;
        [SerializeField, Min(1)] private int maxPinnedClips = 10;

        private sealed class AvatarState
        {
            public Avatar Avatar;
            public FloatingAvatar Floating;
            public string Uuid;
            public GameObject Proxy;
            public Transform[] Sources;
            public Transform[] Targets;
            public bool[] Roots;
            public int HeadIndex;
            public Renderer[] BodyRenderers;
            public Renderer[] ProxyRenderers;
            public Renderer[] HiddenRenderers;
            public bool[] PriorForceRenderingOff;
            public int[] BodyHiddenIndices;
            public Animator[] Animators;
            public AnimatorCullingMode[] PriorCulling;
            public SkinnedMeshRenderer[] Skins;
            public bool[] PriorUpdateWhenOffscreen;
            public MotionPoseBuffer History;
            public MotionPoseFrame Live;
            public MotionPoseFrame Displayed;
            public MotionPoseFrame Sample;
            public MotionPoseFrame BlendFrom;
            public string ActiveRequest;
            public MotionPoseClip ActiveClip;
            public double Progress;
            public double NextCapture;
            public double BlendStarted;
            public bool BeginPresentation;
            public bool Releasing;
            public bool Visible;
        }

        private sealed class Pinned
        {
            public AvatarState State;
            public MotionPoseClip Clip;
        }

        private readonly Dictionary<string, AvatarState> avatars = new Dictionary<string, AvatarState>();
        private readonly Dictionary<string, Pinned> pinned = new Dictionary<string, Pinned>();
        private readonly List<string> invalidAvatars = new List<string>();
        private AvatarManager manager;
        private RoomClient room;
        private Transform roomRoot;
        private bool subscribed;

        public void Initialize(AvatarManager avatarManager, RoomClient roomClient)
        {
            Unsubscribe();
            manager = avatarManager;
            room = roomClient;
            var scene = manager ? NetworkScene.Find(manager) : null;
            roomRoot = scene ? scene.transform : null;
            Subscribe();
        }

        /// <summary>Copies the complete original sentence interval from the remote rolling buffer.</summary>
        public bool Pin(string requestId, string speakerUuid, double startSeconds, double endSeconds, out string reason)
        {
            reason = null;
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(speakerUuid))
            {
                reason = "Missing sentence or speaker identity.";
                return false;
            }
            if (IsLocal(speakerUuid))
            {
                reason = "Local avatars are never recorded or replayed.";
                return false;
            }
            if (pinned.TryGetValue(requestId, out var existing))
            {
                if (existing.State.Uuid == speakerUuid) return true;
                reason = "Sentence identity belongs to another speaker.";
                return false;
            }
            if (!avatars.TryGetValue(speakerUuid, out var state) || !IsRemote(state.Avatar))
            {
                reason = "No supported remote avatar is available.";
                return false;
            }
            if (pinned.Count >= maxPinnedClips)
            {
                reason = "Pinned motion queue is full.";
                return false;
            }
            if (!state.History.TryPin(startSeconds, endSeconds, out var clip))
            {
                reason = "The complete original sentence is not in the motion history.";
                return false;
            }
            pinned.Add(requestId, new Pinned { State = state, Clip = clip });
            return true;
        }

        /// <summary>Sets the motion position from consumed translated-audio samples, including a stationary underrun.</summary>
        public void Present(string requestId, string speakerUuid, double normalizedProgress)
        {
            if (string.IsNullOrEmpty(requestId) || IsLocal(speakerUuid) ||
                !pinned.TryGetValue(requestId, out var item) || item.State.Uuid != speakerUuid ||
                !IsRemote(item.State.Avatar)) return;
            var state = item.State;
            if (state.ActiveRequest != requestId || state.Releasing)
            {
                state.ActiveRequest = requestId;
                state.ActiveClip = item.Clip;
                state.Releasing = false;
                state.BeginPresentation = true;
            }
            state.Progress = double.IsNaN(normalizedProgress) || double.IsInfinity(normalizedProgress)
                ? 0 : Math.Max(0, Math.Min(1, normalizedProgress));
        }

        public void Finish(string requestId)
        {
            if (string.IsNullOrEmpty(requestId) || !pinned.TryGetValue(requestId, out var item)) return;
            pinned.Remove(requestId);
            var state = item.State;
            if (state.ActiveRequest != requestId) return;
            state.ActiveRequest = null;
            state.ActiveClip = null;
            state.BeginPresentation = false;
            if (!state.Visible) return;
            state.BlendFrom.CopyFrom(state.Displayed);
            state.BlendStarted = Time.realtimeSinceStartupAsDouble;
            state.Releasing = true;
        }

        public void ResetReplay()
        {
            pinned.Clear();
            foreach (var state in avatars.Values)
            {
                HideProxy(state);
                state.History.Clear();
                state.NextCapture = 0;
                state.ActiveRequest = null;
                state.ActiveClip = null;
                state.BeginPresentation = false;
                state.Releasing = false;
            }
        }

        /// <summary>Reads the displayed remote head or a live head, including the local head for audio positioning only.</summary>
        public bool TryGetHead(string speakerUuid, out Pose pose)
        {
            pose = default;
            if (string.IsNullOrEmpty(speakerUuid)) return false;
            if (avatars.TryGetValue(speakerUuid, out var state) && IsRemote(state.Avatar))
            {
                var head = state.Visible ? state.Targets[state.HeadIndex] : state.Floating.head;
                if (!head) return false;
                pose = new Pose(head.position, head.rotation);
                return true;
            }
            if (!manager) return false;
            foreach (var avatar in manager.Avatars)
            {
                if (!avatar || avatar.Peer == null || avatar.Peer.uuid != speakerUuid) continue;
                var floating = avatar.GetComponentInChildren<FloatingAvatar>(true);
                if (!floating || !floating.head) return false;
                pose = new Pose(floating.head.position, floating.head.rotation);
                return true;
            }
            return false;
        }

        public void CancelSpeaker(string uuid)
        {
            if (string.IsNullOrEmpty(uuid)) return;
            var requests = new List<string>();
            foreach (var entry in pinned)
                if (entry.Value.State.Uuid == uuid) requests.Add(entry.Key);
            foreach (var request in requests) pinned.Remove(request);
            if (!avatars.TryGetValue(uuid, out var state)) return;
            HideProxy(state);
            state.History.Clear();
            state.ActiveRequest = null;
            state.ActiveClip = null;
            state.BeginPresentation = false;
            state.Releasing = false;
        }

        private bool IsLocal(string uuid) => room && room.Me != null && room.Me.uuid == uuid;
        private bool IsRemote(Avatar avatar) => avatar && !avatar.IsLocal && avatar.Peer != null &&
            !string.IsNullOrEmpty(avatar.Peer.uuid) && !IsLocal(avatar.Peer.uuid);

        private void OnEnable() => Subscribe();

        private void Subscribe()
        {
            if (subscribed || !isActiveAndEnabled || !manager || !room || !roomRoot) return;
            subscribed = true;
            manager.OnAvatarCreated.AddListener(OnAvatarCreated);
            manager.OnAvatarDestroyed.AddListener(OnAvatarDestroyed);
        }

        private void OnAvatarCreated(Avatar avatar)
        {
            if (!IsRemote(avatar)) return;
            var floating = avatar.GetComponentInChildren<FloatingAvatar>(true);
            if (!floating || !floating.head || !floating.torso || !floating.leftHand || !floating.rightHand ||
                !floating.headRenderer || !floating.torsoRenderer || !floating.leftHandRenderer || !floating.rightHandRenderer)
                return;
            var uuid = avatar.Peer.uuid;
            if (avatars.TryGetValue(uuid, out var previous))
            {
                if (previous.Avatar == avatar) return;
                CancelSpeaker(uuid);
                DisposeState(previous);
                avatars.Remove(uuid);
            }
            var state = new AvatarState { Avatar = avatar, Floating = floating, Uuid = uuid };
            try
            {
                BuildProxy(state);
                var count = state.Sources.Length;
                state.History = new MotionPoseBuffer(historySeconds, captureRate, count, 4);
                state.Live = new MotionPoseFrame(count, 4);
                state.Displayed = new MotionPoseFrame(count, 4);
                state.Sample = new MotionPoseFrame(count, 4);
                state.BlendFrom = new MotionPoseFrame(count, 4);
                KeepLiveAnimationUpdating(state);
                avatars.Add(uuid, state);
            }
            catch (Exception error)
            {
                DisposeState(state);
                Debug.LogWarning($"[Translation motion] Remote avatar cannot be replayed: {error.Message}", this);
            }
        }

        private void OnAvatarDestroyed(Avatar avatar)
        {
            if (!avatar || avatar.Peer == null || string.IsNullOrEmpty(avatar.Peer.uuid)) return;
            var uuid = avatar.Peer.uuid;
            if (!avatars.TryGetValue(uuid, out var state) || state.Avatar != avatar) return;
            CancelSpeaker(uuid);
            DisposeState(state);
            avatars.Remove(uuid);
        }

        private void LateUpdate()
        {
            if (!subscribed || !roomRoot) return;
            var now = Time.realtimeSinceStartupAsDouble;
            invalidAvatars.Clear();
            foreach (var state in avatars.Values)
            {
                // Recheck before every capture/apply, even if an avatar's role changed after binding.
                if (!IsRemote(state.Avatar))
                {
                    invalidAvatars.Add(state.Uuid);
                    continue;
                }
                CaptureLive(state, now);
                if (now >= state.NextCapture)
                {
                    state.History.Add(now, state.Live);
                    var interval = 1.0 / Math.Max(1, captureRate);
                    state.NextCapture = state.NextCapture > 0 ? state.NextCapture + interval : now + interval;
                    if (state.NextCapture <= now) state.NextCapture = now + interval;
                }
                if (state.BeginPresentation)
                {
                    state.BlendFrom.CopyFrom(state.Visible ? state.Displayed : state.Live);
                    state.BlendStarted = now;
                    state.BeginPresentation = false;
                    ShowProxy(state);
                }
                if (state.ActiveClip != null && state.ActiveRequest != null)
                {
                    state.ActiveClip.Sample(state.Progress, state.Sample);
                    MotionPoseFrame.Interpolate(state.BlendFrom, state.Sample, BlendWeight(now, state.BlendStarted), state.Displayed);
                    Apply(state);
                }
                else if (state.Releasing)
                {
                    var weight = BlendWeight(now, state.BlendStarted);
                    MotionPoseFrame.Interpolate(state.BlendFrom, state.Live, weight, state.Displayed);
                    Apply(state);
                    if (weight >= 1)
                    {
                        state.Releasing = false;
                        HideProxy(state);
                    }
                }
            }
            foreach (var uuid in invalidAvatars)
            {
                CancelSpeaker(uuid);
                DisposeState(avatars[uuid]);
                avatars.Remove(uuid);
            }
        }

        private double BlendWeight(double now, double started) =>
            blendSeconds > 0 ? Math.Max(0, Math.Min(1, (now - started) / blendSeconds)) : 1;

        private void CaptureLive(AvatarState state, double now)
        {
            var inverseRotation = Quaternion.Inverse(roomRoot.rotation);
            var roomScale = roomRoot.lossyScale;
            state.Live.Time = now;
            for (var i = 0; i < state.Sources.Length; ++i)
            {
                var source = state.Sources[i];
                if (state.Roots[i])
                {
                    state.Live.Positions[i] = roomRoot.InverseTransformPoint(source.position);
                    state.Live.Rotations[i] = inverseRotation * source.rotation;
                    var scale = source.lossyScale;
                    state.Live.Scales[i] = new Vector3(SafeDivide(scale.x, roomScale.x),
                        SafeDivide(scale.y, roomScale.y), SafeDivide(scale.z, roomScale.z));
                }
                else
                {
                    state.Live.Positions[i] = source.localPosition;
                    state.Live.Rotations[i] = source.localRotation;
                    state.Live.Scales[i] = source.localScale;
                }
            }
            for (var i = 0; i < state.BodyRenderers.Length; ++i)
            {
                var wasHidden = state.Visible
                    ? state.PriorForceRenderingOff[state.BodyHiddenIndices[i]]
                    : state.BodyRenderers[i].forceRenderingOff;
                state.Live.Visible[i] = !wasHidden && state.BodyRenderers[i].enabled &&
                    state.BodyRenderers[i].gameObject.activeInHierarchy;
            }
        }

        private static float SafeDivide(float numerator, float denominator) =>
            Mathf.Abs(denominator) > 0.000001f ? numerator / denominator : numerator;

        private static void Apply(AvatarState state)
        {
            if (!state.Visible) return;
            for (var i = 0; i < state.Targets.Length; ++i)
            {
                state.Targets[i].localPosition = state.Displayed.Positions[i];
                state.Targets[i].localRotation = state.Displayed.Rotations[i];
                state.Targets[i].localScale = state.Displayed.Scales[i];
            }
            for (var i = 0; i < state.ProxyRenderers.Length; ++i)
                state.ProxyRenderers[i].enabled = state.Displayed.Visible[i];
        }

        private static void ShowProxy(AvatarState state)
        {
            if (state.Visible) return;
            for (var i = 0; i < state.HiddenRenderers.Length; ++i)
            {
                if (!state.HiddenRenderers[i]) continue;
                state.PriorForceRenderingOff[i] = state.HiddenRenderers[i].forceRenderingOff;
                state.HiddenRenderers[i].forceRenderingOff = true;
            }
            // The latest texture choice is applied only when presentation starts, without material instantiation.
            for (var i = 0; i < state.BodyRenderers.Length; ++i)
                state.ProxyRenderers[i].sharedMaterials = state.BodyRenderers[i].sharedMaterials;
            state.Visible = true;
            state.Proxy.SetActive(true);
        }

        private static void HideProxy(AvatarState state)
        {
            if (state.Visible && state.HiddenRenderers != null)
                for (var i = 0; i < state.HiddenRenderers.Length; ++i)
                    if (state.HiddenRenderers[i]) state.HiddenRenderers[i].forceRenderingOff = state.PriorForceRenderingOff[i];
            state.Visible = false;
            if (state.Proxy) state.Proxy.SetActive(false);
        }

        private void BuildProxy(AvatarState state)
        {
            state.Proxy = new GameObject("Translation motion " + state.Uuid);
            state.Proxy.SetActive(false);
            state.Proxy.transform.SetParent(roomRoot, false);
            var map = new Dictionary<Transform, Transform>();
            var sources = new List<Transform>();
            var targets = new List<Transform>();
            var roots = new List<bool>();
            var body = state.Floating;
            var sourceRoots = new[] { body.head, body.torso, body.leftHand, body.rightHand };
            foreach (var sourceRoot in sourceRoots)
                CopyTransforms(sourceRoot, state.Proxy.transform, true, map, sources, targets, roots);
            state.Sources = sources.ToArray();
            state.Targets = targets.ToArray();
            state.Roots = roots.ToArray();
            state.HeadIndex = sources.IndexOf(body.head);
            state.BodyRenderers = new[] { body.headRenderer, body.torsoRenderer, body.leftHandRenderer, body.rightHandRenderer };
            state.ProxyRenderers = new Renderer[4];
            for (var i = 0; i < state.BodyRenderers.Length; ++i)
                state.ProxyRenderers[i] = CopyRenderer(state.BodyRenderers[i], map);
            state.HiddenRenderers = state.Avatar.GetComponentsInChildren<Renderer>(true);
            state.PriorForceRenderingOff = new bool[state.HiddenRenderers.Length];
            state.BodyHiddenIndices = new int[4];
            for (var i = 0; i < state.BodyRenderers.Length; ++i)
            {
                state.BodyHiddenIndices[i] = Array.IndexOf(state.HiddenRenderers, state.BodyRenderers[i]);
                if (state.BodyHiddenIndices[i] < 0)
                    throw new InvalidOperationException("A body renderer does not belong to this avatar.");
            }
        }

        private static void CopyTransforms(Transform source, Transform parent, bool root,
            Dictionary<Transform, Transform> map, List<Transform> sources, List<Transform> targets, List<bool> roots)
        {
            if (map.ContainsKey(source)) throw new InvalidOperationException("Overlapping avatar visual roots.");
            var copy = new GameObject(source.name).transform;
            copy.gameObject.layer = source.gameObject.layer;
            copy.SetParent(parent, false);
            copy.localPosition = source.localPosition;
            copy.localRotation = source.localRotation;
            copy.localScale = source.localScale;
            map.Add(source, copy);
            sources.Add(source);
            targets.Add(copy);
            roots.Add(root);
            for (var i = 0; i < source.childCount; ++i)
                CopyTransforms(source.GetChild(i), copy, false, map, sources, targets, roots);
        }

        private static Renderer CopyRenderer(Renderer source, Dictionary<Transform, Transform> map)
        {
            if (!map.TryGetValue(source.transform, out var target))
                throw new InvalidOperationException("A body renderer is outside its visual hierarchy.");
            Renderer destination;
            if (source is SkinnedMeshRenderer skin)
            {
                var clone = target.gameObject.AddComponent<SkinnedMeshRenderer>();
                clone.sharedMesh = skin.sharedMesh;
                var sourceBones = skin.bones;
                var bones = new Transform[sourceBones.Length];
                for (var i = 0; i < bones.Length; ++i)
                {
                    if (!sourceBones[i] || !map.TryGetValue(sourceBones[i], out bones[i]))
                        throw new InvalidOperationException("A hand bone is outside its visual hierarchy.");
                }
                clone.bones = bones;
                if (skin.rootBone)
                {
                    if (!map.TryGetValue(skin.rootBone, out var rootBone))
                        throw new InvalidOperationException("A hand root bone is outside its visual hierarchy.");
                    clone.rootBone = rootBone;
                }
                clone.localBounds = skin.localBounds;
                clone.quality = skin.quality;
                clone.updateWhenOffscreen = true;
                clone.skinnedMotionVectors = skin.skinnedMotionVectors;
                if (skin.sharedMesh)
                    for (var i = 0; i < skin.sharedMesh.blendShapeCount; ++i)
                        clone.SetBlendShapeWeight(i, skin.GetBlendShapeWeight(i));
                destination = clone;
            }
            else if (source is MeshRenderer)
            {
                var filter = source.GetComponent<MeshFilter>();
                if (!filter) throw new InvalidOperationException("A body mesh filter is missing.");
                target.gameObject.AddComponent<MeshFilter>().sharedMesh = filter.sharedMesh;
                destination = target.gameObject.AddComponent<MeshRenderer>();
            }
            else throw new InvalidOperationException("Unsupported avatar renderer.");
            destination.sharedMaterials = source.sharedMaterials;
            destination.shadowCastingMode = source.shadowCastingMode;
            destination.receiveShadows = source.receiveShadows;
            destination.lightProbeUsage = source.lightProbeUsage;
            destination.reflectionProbeUsage = source.reflectionProbeUsage;
            destination.motionVectorGenerationMode = source.motionVectorGenerationMode;
            destination.renderingLayerMask = source.renderingLayerMask;
            destination.sortingLayerID = source.sortingLayerID;
            destination.sortingOrder = source.sortingOrder;
            destination.enabled = source.enabled;
            var properties = new MaterialPropertyBlock();
            source.GetPropertyBlock(properties);
            destination.SetPropertyBlock(properties);
            return destination;
        }

        private static void KeepLiveAnimationUpdating(AvatarState state)
        {
            state.Animators = state.Avatar.GetComponentsInChildren<Animator>(true);
            state.PriorCulling = new AnimatorCullingMode[state.Animators.Length];
            for (var i = 0; i < state.Animators.Length; ++i)
            {
                state.PriorCulling[i] = state.Animators[i].cullingMode;
                state.Animators[i].cullingMode = AnimatorCullingMode.AlwaysAnimate;
            }
            state.Skins = state.Avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            state.PriorUpdateWhenOffscreen = new bool[state.Skins.Length];
            for (var i = 0; i < state.Skins.Length; ++i)
            {
                state.PriorUpdateWhenOffscreen[i] = state.Skins[i].updateWhenOffscreen;
                state.Skins[i].updateWhenOffscreen = true;
            }
        }

        private static void DisposeState(AvatarState state)
        {
            HideProxy(state);
            if (state.Animators != null)
                for (var i = 0; i < state.Animators.Length; ++i)
                    if (state.Animators[i]) state.Animators[i].cullingMode = state.PriorCulling[i];
            if (state.Skins != null)
                for (var i = 0; i < state.Skins.Length; ++i)
                    if (state.Skins[i]) state.Skins[i].updateWhenOffscreen = state.PriorUpdateWhenOffscreen[i];
            if (state.Proxy)
            {
                if (Application.isPlaying) Destroy(state.Proxy);
                else DestroyImmediate(state.Proxy);
            }
        }

        private void Unsubscribe()
        {
            if (subscribed && manager)
            {
                manager.OnAvatarCreated.RemoveListener(OnAvatarCreated);
                manager.OnAvatarDestroyed.RemoveListener(OnAvatarDestroyed);
            }
            subscribed = false;
            pinned.Clear();
            foreach (var state in avatars.Values) DisposeState(state);
            avatars.Clear();
        }

        private void OnDisable() => Unsubscribe();
        private void OnDestroy() => Unsubscribe();
    }
}
