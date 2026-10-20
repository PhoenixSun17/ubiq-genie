using System;
using NUnit.Framework;
using Ubiq.Genie.Samples.RealtimeTranslation;

namespace UbiqGenie.RealtimeTranslation.Tests
{
    public sealed class SoVITSVoiceSelectionTests
    {
        private const string Peer = "11111111-1111-4111-8111-111111111111";
        private const string Backend = "22222222-2222-4222-8222-222222222222";
        private const string Room = "6765c52b-3ad6-4fb0-9030-2c9a05dc4731";
        private const string Request = "33333333-3333-4333-8333-333333333333";
        private static readonly string VoiceA = new string('a', 64);
        private static readonly string VoiceB = new string('b', 64);

        private static SoVITSVoiceOption[] Voices() => new[]
        {
            new SoVITSVoiceOption { id = VoiceA, name = "reference-a.wav", language = "en" },
            new SoVITSVoiceOption { id = VoiceB, name = "reference-b.wav", language = "zh" }
        };

        private static SoVITSVoiceSelectionState State(string remembered = "")
        {
            var state = new SoVITSVoiceSelectionState();
            state.SetSession(Peer, Room, remembered);
            state.SetBackend(Backend, Voices());
            return state;
        }

        private static SoVITSVoiceBinding Ack(string status = "bound") => new SoVITSVoiceBinding
        {
            peerUuid = Peer, requestId = Request, voiceId = VoiceA, status = status, message = "Reference unavailable."
        };

        [Test]
        public void FirstUseDoesNotChooseTheOnlyOrFirstVoice()
        {
            var state = State();
            Assert.That(state.SelectedVoiceId, Is.Empty);
            Assert.That(state.ShouldRestore, Is.False);
            Assert.That(state.CanApply, Is.False);
            Assert.Throws<InvalidOperationException>(() => state.BeginRequest(Request));
        }

        [Test]
        public void ManualChoiceIsRememberedOnlyAfterMatchingConfirmation()
        {
            var state = State();
            state.Choose(VoiceA);
            var request = state.BeginRequest(Request);
            Assert.That(request.backendUuid, Is.EqualTo(Backend));
            Assert.That(request.voiceId, Is.EqualTo(VoiceA));
            Assert.That(state.RememberedVoiceId, Is.Empty);
            Assert.That(state.BoundVoiceId, Is.Empty);
            Assert.That(state.AwaitingConfirmation, Is.True);
            Assert.That(state.Acknowledge(Backend, Ack()), Is.True);
            Assert.That(state.RememberedVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.BoundVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False, "Duplicate acknowledgements must not write preferences again.");
        }

        [TestCase("backend")]
        [TestCase("peer")]
        [TestCase("request")]
        [TestCase("voice")]
        [TestCase("status")]
        public void UnrelatedOrInvalidAcknowledgementsNeverConfirm(string mismatch)
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            var ack = Ack();
            var sender = Backend;
            switch (mismatch)
            {
                case "backend": sender = Peer; break;
                case "peer": ack.peerUuid = Backend; break;
                case "request": ack.requestId = Guid.NewGuid().ToString(); break;
                case "voice": ack.voiceId = VoiceB; break;
                case "status": ack.status = "ready"; break;
            }
            Assert.That(state.Acknowledge(sender, ack), Is.False);
            Assert.That(state.BoundVoiceId, Is.Empty);
            Assert.That(state.AwaitingConfirmation, Is.True);
        }

        [Test]
        public void ActiveProfileErrorRevokesConfirmationButRetainsRememberedChoice()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.Acknowledge(Backend, Ack());
            Assert.That(state.Acknowledge(Backend, Ack("error")), Is.False);
            Assert.That(state.BoundVoiceId, Is.Empty);
            Assert.That(state.Error, Does.Contain("unavailable"));
            Assert.That(state.RememberedVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.ShouldRestore, Is.False, "An error must not cause an automatic request loop.");
        }

        [Test]
        public void RejectedNewChoicePreservesThePreviousEffectiveAndRememberedVoice()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.Acknowledge(Backend, Ack());
            state.Choose(VoiceB);
            Assert.That(state.BoundVoiceId, Is.EqualTo(VoiceA), "Choosing in the Inspector does not change the effective binding.");
            var request = state.BeginRequest(Guid.NewGuid().ToString());
            Assert.That(state.BoundVoiceId, Is.EqualTo(VoiceA), "The previous voice remains effective while awaiting confirmation.");
            var error = new SoVITSVoiceBinding
            {
                peerUuid = Peer, requestId = request.requestId, voiceId = VoiceB,
                status = "error", message = "The new reference is unavailable."
            };
            Assert.That(state.Acknowledge(Backend, error), Is.False);
            Assert.That(state.BoundVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.RememberedVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.Error, Does.Contain("unavailable"));
        }

        [Test]
        public void NewChoiceReplacesTheEffectiveVoiceOnlyAfterItsMatchingAcknowledgement()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.Acknowledge(Backend, Ack());
            state.Choose(VoiceB);
            var request = state.BeginRequest(Guid.NewGuid().ToString());
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False);
            Assert.That(state.BoundVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.Acknowledge(Backend, new SoVITSVoiceBinding
            {
                peerUuid = Peer, requestId = request.requestId, voiceId = VoiceB, status = "bound"
            }), Is.True);
            Assert.That(state.BoundVoiceId, Is.EqualTo(VoiceB));
            Assert.That(state.RememberedVoiceId, Is.EqualTo(VoiceB));
        }

        [Test]
        public void BackendRestartRestoresRememberedChoiceWithANewTargetAndRequest()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.Acknowledge(Backend, Ack());
            state.SetBackendProblem("No backend.");
            Assert.That(state.CanApply, Is.False);
            Assert.That(state.BoundVoiceId, Is.Empty);
            state.SetBackend(Peer, Voices());
            Assert.That(state.ShouldRestore, Is.True);
            var request = state.BeginRequest(Guid.NewGuid().ToString());
            Assert.That(request.backendUuid, Is.EqualTo(Peer));
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False);
            Assert.That(state.ShouldRestore, Is.False);
        }

        [Test]
        public void MultipleBackendsInvalidatePendingConfirmationUntilOneRemains()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.SetBackendProblem("Multiple translation backends.");
            Assert.That(state.Status, Does.Contain("Multiple"));
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False);
            Assert.That(state.ShouldRestore, Is.False);
            state.SetBackend(Backend, Voices());
            Assert.That(state.ShouldRestore, Is.True);
        }

        [Test]
        public void MissingRememberedVoiceWaitsForAvailabilityWithoutSelectingAnother()
        {
            var state = State(VoiceA);
            state.SetBackend(Backend, new[] { Voices()[1] });
            Assert.That(state.SelectedVoiceId, Is.EqualTo(VoiceA));
            Assert.That(state.Status, Does.Contain("unavailable"));
            Assert.That(state.ShouldRestore, Is.False);
            state.SetBackend(Backend, Voices());
            Assert.That(state.ShouldRestore, Is.True);
        }

        [Test]
        public void DeletedCatalogEntryCannotBeConfirmedByAnOldBoundReply()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.SetBackend(Backend, new[] { Voices()[1] });
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False);
            Assert.That(state.BoundVoiceId, Is.Empty);
            Assert.That(state.Status, Does.Contain("unavailable"));
        }

        [Test]
        public void NewSessionUuidRejectsOldRepliesAndRestoresSavedVoice()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.SetSession(Backend, Room, VoiceA);
            state.SetBackend(Backend, Voices());
            Assert.That(state.ShouldRestore, Is.True);
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False);
            Assert.That(state.PeerUuid, Is.EqualTo(Backend));
        }

        [Test]
        public void ChangingRoomLoadsOnlyThatRoomsRememberedChoice()
        {
            var state = State(VoiceA);
            state.SetSession(Peer, Backend, "");
            state.SetBackend(Backend, Voices());
            Assert.That(state.SelectedVoiceId, Is.Empty);
            Assert.That(state.ShouldRestore, Is.False);
        }

        [Test]
        public void ManualChoiceOverridesRestoreWithoutSavingAnUnconfirmedVoice()
        {
            var state = State(VoiceA);
            Assert.That(state.Choose(VoiceB), Is.True);
            Assert.That(state.ShouldRestore, Is.False);
            Assert.That(state.RememberedVoiceId, Is.EqualTo(VoiceA));
            state.BeginRequest(Request);
            Assert.That(state.Choose(VoiceA), Is.False, "Keep the pending request's selection stable.");
        }

        [Test]
        public void TimeoutAllowsRetryAndNewRequestRejectsLateOldAcknowledgement()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.ExpireRequest();
            Assert.That(state.Error, Does.Contain("15 seconds"));
            Assert.That(state.CanApply, Is.True);
            Assert.That(state.ShouldRestore, Is.False);
            state.BeginRequest(Guid.NewGuid().ToString());
            Assert.That(state.Acknowledge(Backend, Ack()), Is.False);
        }

        [Test]
        public void MatchingLateAcknowledgementCanConfirmWithoutANewerRequest()
        {
            var state = State(VoiceA);
            state.BeginRequest(Request);
            state.ExpireRequest();
            Assert.That(state.Acknowledge(Backend, Ack()), Is.True);
            Assert.That(state.Error, Is.Empty);
        }

        [Test]
        public void CatalogRejectsDuplicateIdsAndUnsupportedVersion()
        {
            var catalog = new SoVITSVoiceCatalog { version = 1, voices = Voices() };
            Assert.That(SoVITSVoiceSelectionState.IsValidCatalog(catalog), Is.True);
            catalog.voices[1].id = VoiceA;
            Assert.That(SoVITSVoiceSelectionState.IsValidCatalog(catalog), Is.False);
            catalog.voices = Voices();
            catalog.version = 2;
            Assert.That(SoVITSVoiceSelectionState.IsValidCatalog(catalog), Is.False);
        }

        [TestCase("")]
        [TestCase("profile.wav")]
        [TestCase("../reference")]
        [TestCase("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")]
        public void VoiceIdsMustBeLowercaseSha256(string id)
        {
            Assert.That(SoVITSVoiceSelectionState.IsVoiceId(id), Is.False);
        }

        [Test]
        public void EmptyCatalogIsValidButOffersNoAutomaticChoice()
        {
            var catalog = new SoVITSVoiceCatalog { version = 1, voices = Array.Empty<SoVITSVoiceOption>() };
            Assert.That(SoVITSVoiceSelectionState.IsValidCatalog(catalog), Is.True);
            var state = State();
            state.SetBackend(Backend, catalog.voices);
            Assert.That(state.Status, Does.Contain("no valid reference voices"));
            Assert.That(state.CanApply, Is.False);
        }
    }
}
