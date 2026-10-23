using System.IO;
using System.Linq;
using NUnit.Framework;
using Ubiq.Avatars;
using Ubiq.Rooms;
using Ubiq.Voip;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion.Tests
{
    public sealed class MotionSceneTests
    {
        private const string Folder = "Assets/Ubiq-Genie/Apps/RealtimeTranslation/";

        [Test]
        public void CopiedSceneHasOneReceiverKeepsLiveNetworkingAndOwnConfigurationAssets()
        {
            var originalSetup = EditorSceneManager.GetSceneManagerSetup();
            var scene = EditorSceneManager.OpenScene(Folder + "RealtimeTranslationMotion.unity", OpenSceneMode.Additive);
            try
            {
                var roots = scene.GetRootGameObjects();
                var components = roots.SelectMany(root => root.GetComponentsInChildren<MonoBehaviour>(true)).ToArray();
                foreach (var transform in roots.SelectMany(root => root.GetComponentsInChildren<Transform>(true)))
                    Assert.Zero(GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(transform.gameObject),
                        $"Missing script on {transform.name}");
                var receiver = components.OfType<MotionTranslationReceiver>().Single();
                Assert.IsTrue(receiver.enabled && receiver.gameObject.activeInHierarchy);
                Assert.NotNull(receiver.GetComponent<RemoteAvatarReplay>());
                Assert.NotNull(receiver.GetComponent<MotionAudioPlayer>());
                Assert.IsTrue(receiver.GetComponent<OriginalVoiceToggle>().playOriginalVoice);
                Assert.IsTrue(components.OfType<AvatarManager>().Any());
                Assert.IsTrue(components.OfType<VoipPeerConnectionManager>().Any());
                Assert.IsTrue(components.Any(item => item && item.GetType().Name == "SoVITSVoiceSelector"));
                Assert.IsFalse(components.Any(item => item && item.isActiveAndEnabled &&
                    (item.GetType().Name == "SoVITSReceiver" || item.GetType().Name == "SoVITSAudioSource")));
                var serialized = new SerializedObject(receiver);
                Assert.IsInstanceOf<RoomClient>(serialized.FindProperty("roomClient").objectReferenceValue);
                string sceneText = File.ReadAllText(Folder + "RealtimeTranslationMotion.unity");
                foreach (string name in new[] { "SoVITSServerConfig", "SoVITSRoomGuid" })
                {
                    string oldPath = Folder + name + ".asset";
                    string newPath = Folder + name + "Motion.asset";
                    Assert.AreNotEqual(AssetDatabase.AssetPathToGUID(oldPath), AssetDatabase.AssetPathToGUID(newPath));
                    Assert.IsFalse(sceneText.Contains(AssetDatabase.AssetPathToGUID(oldPath)));
                    StringAssert.Contains(AssetDatabase.AssetPathToGUID(newPath), sceneText);
                    // Asset identity/name is distinct while the existing server/room settings stay identical.
                    Assert.AreEqual(File.ReadAllText(oldPath).Replace("\r\n", "\n"),
                        File.ReadAllText(newPath).Replace(name + "Motion", name).Replace("\r\n", "\n"));
                }
            }
            finally
            {
                EditorSceneManager.CloseScene(scene, true);
                if (originalSetup.Any(item => item.isLoaded && item.isActive))
                    EditorSceneManager.RestoreSceneManagerSetup(originalSetup);
            }
        }
    }
}
