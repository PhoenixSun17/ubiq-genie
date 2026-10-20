using Ubiq.Genie.Samples.RealtimeTranslation;
using UnityEditor;
using UnityEngine;

namespace Ubiq.Genie.Editor
{
    [CustomEditor(typeof(SoVITSVoiceSelector))]
    public sealed class SoVITSVoiceSelectorEditor : UnityEditor.Editor
    {
        public override bool RequiresConstantRepaint() => Application.isPlaying;

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            using (new EditorGUI.DisabledScope(Application.isPlaying))
                EditorGUILayout.PropertyField(serializedObject.FindProperty("roomClient"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("restorePreviousVoice"));
            serializedObject.ApplyModifiedProperties();
            var selector = (SoVITSVoiceSelector)target;
            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox("Enter Play mode and join the room. The Node app supplies the voice list.", MessageType.Info);
                return;
            }

            var state = selector.State;
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Local UUID", state.PeerUuid.Length > 0 ? state.PeerUuid : "Unavailable");
            EditorGUILayout.LabelField("Room UUID", state.RoomUuid.Length > 0 ? state.RoomUuid : "Not joined");
            EditorGUILayout.LabelField("Backend UUID", state.BackendUuid.Length > 0 ? state.BackendUuid : "Unavailable");
            EditorGUILayout.LabelField("Remembered voice", NameForId(state, state.RememberedVoiceId));
            EditorGUILayout.LabelField("Confirmed voice", NameForId(state, state.BoundVoiceId));

            var labels = new string[state.Voices.Count + 1];
            labels[0] = state.SelectedVoiceId.Length > 0 && !state.Contains(state.SelectedVoiceId) ?
                "Remembered voice unavailable — choose a voice" : "Choose a voice...";
            var selected = 0;
            for (var i = 0; i < state.Voices.Count; ++i)
            {
                var voice = state.Voices[i];
                labels[i + 1] = $"{voice.name} ({voice.language})";
                if (voice.id == state.SelectedVoiceId) selected = i + 1;
            }
            using (new EditorGUI.DisabledScope(!selector.isActiveAndEnabled || state.AwaitingConfirmation || state.Voices.Count == 0))
            {
                var next = EditorGUILayout.Popup("Voice", selected, labels);
                if (next != selected) selector.ChooseVoice(next == 0 ? "" : state.Voices[next - 1].id);
            }
            using (new EditorGUI.DisabledScope(!selector.isActiveAndEnabled || !state.CanApply))
            {
                if (GUILayout.Button("Apply Voice")) selector.ApplySelectedVoice();
            }
            using (new EditorGUI.DisabledScope(!selector.isActiveAndEnabled))
            {
                if (GUILayout.Button("Refresh backend status")) selector.Refresh();
            }
            EditorGUILayout.HelpBox(selector.Status, state.Error.Length > 0 ? MessageType.Warning : MessageType.Info);
        }

        private static string NameForId(SoVITSVoiceSelectionState state, string id)
        {
            if (string.IsNullOrEmpty(id)) return "None";
            foreach (var voice in state.Voices) if (voice.id == id) return voice.name;
            return "Unavailable (" + id.Substring(0, 12) + "...)";
        }
    }
}
