using System;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Maps Node performance.now milliseconds to the client's monotonic seconds.</summary>
    public sealed class MotionClock
    {
        private readonly double[] offsets = new double[8];
        private readonly double[] roundTrips = new double[8];
        private readonly double[] receivedAt = new double[8];
        private int count;
        private int next;

        public bool Observe(double t0, double s1, double s2, double t3)
        {
            if (!Finite(t0) || !Finite(s1) || !Finite(s2) || !Finite(t3) ||
                t3 < t0 || s2 < s1) return false;
            double roundTrip = (t3 - t0) - (s2 - s1);
            if (roundTrip < -0.01 || roundTrip > 10000) return false;
            offsets[next] = ((s1 - t0) + (s2 - t3)) / 2;
            roundTrips[next] = Math.Max(0, roundTrip);
            receivedAt[next] = t3;
            next = (next + 1) % offsets.Length;
            count = Math.Min(count + 1, offsets.Length);
            return true;
        }

        public bool TryMap(double serverMilliseconds, double nowClientMilliseconds, out double clientSeconds)
        {
            clientSeconds = 0;
            if (!Finite(serverMilliseconds) || !Finite(nowClientMilliseconds)) return false;
            int best = -1;
            for (int i = 0; i < count; ++i)
            {
                double age = nowClientMilliseconds - receivedAt[i];
                if (age < 0 || age > 15000) continue;
                if (best < 0 || roundTrips[i] < roundTrips[best]) best = i;
            }
            if (best < 0) return false;
            clientSeconds = (serverMilliseconds - offsets[best]) / 1000;
            return true;
        }

        public void Clear()
        {
            count = next = 0;
        }

        public static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
