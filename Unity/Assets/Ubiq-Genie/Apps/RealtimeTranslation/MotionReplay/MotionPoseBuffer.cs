using System;
using UnityEngine;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>A reusable pose snapshot. The owning renderer defines the coordinate space of each transform.</summary>
    public sealed class MotionPoseFrame
    {
        public double Time { get; internal set; }
        public Vector3[] Positions { get; }
        public Quaternion[] Rotations { get; }
        public Vector3[] Scales { get; }
        public bool[] Visible { get; }

        public MotionPoseFrame(int transformCount, int rendererCount)
        {
            if (transformCount < 1 || rendererCount < 0) throw new ArgumentOutOfRangeException();
            Positions = new Vector3[transformCount];
            Rotations = new Quaternion[transformCount];
            Scales = new Vector3[transformCount];
            Visible = new bool[rendererCount];
            for (var i = 0; i < transformCount; ++i)
            {
                Rotations[i] = Quaternion.identity;
                Scales[i] = Vector3.one;
            }
        }

        public void CopyFrom(MotionPoseFrame source)
        {
            ValidateShape(source);
            Time = source.Time;
            Array.Copy(source.Positions, Positions, Positions.Length);
            Array.Copy(source.Rotations, Rotations, Rotations.Length);
            Array.Copy(source.Scales, Scales, Scales.Length);
            Array.Copy(source.Visible, Visible, Visible.Length);
        }

        public static void Interpolate(MotionPoseFrame from, MotionPoseFrame to, double amount, MotionPoseFrame destination)
        {
            destination.ValidateShape(from);
            destination.ValidateShape(to);
            var weight = Mathf.Clamp01((float)amount);
            destination.Time = from.Time + (to.Time - from.Time) * weight;
            for (var i = 0; i < destination.Positions.Length; ++i)
            {
                destination.Positions[i] = Vector3.LerpUnclamped(from.Positions[i], to.Positions[i], weight);
                destination.Rotations[i] = Quaternion.SlerpUnclamped(from.Rotations[i], to.Rotations[i], weight);
                destination.Scales[i] = Vector3.LerpUnclamped(from.Scales[i], to.Scales[i], weight);
            }
            for (var i = 0; i < destination.Visible.Length; ++i)
                destination.Visible[i] = weight < 1 ? from.Visible[i] : to.Visible[i];
        }

        private void ValidateShape(MotionPoseFrame other)
        {
            if (other == null || Positions.Length != other.Positions.Length || Visible.Length != other.Visible.Length)
                throw new ArgumentException("Pose snapshot dimensions differ.");
        }
    }

    /// <summary>An independently copied sentence, unaffected by subsequent rolling-history writes.</summary>
    public sealed class MotionPoseClip
    {
        private readonly MotionPoseFrame[] frames;
        public double StartTime => frames[0].Time;
        public double EndTime => frames[frames.Length - 1].Time;
        public int FrameCount => frames.Length;

        internal MotionPoseClip(MotionPoseFrame[] frames)
        {
            this.frames = frames;
        }

        public void Sample(double normalizedProgress, MotionPoseFrame destination)
        {
            if (double.IsNaN(normalizedProgress) || double.IsInfinity(normalizedProgress))
                normalizedProgress = 0;
            var time = StartTime + (EndTime - StartTime) * Math.Max(0, Math.Min(1, normalizedProgress));
            var low = 0;
            var high = frames.Length - 1;
            while (low + 1 < high)
            {
                var middle = (low + high) / 2;
                if (frames[middle].Time <= time) low = middle;
                else high = middle;
            }
            var duration = frames[high].Time - frames[low].Time;
            MotionPoseFrame.Interpolate(frames[low], frames[high],
                duration > 0 ? (time - frames[low].Time) / duration : 0, destination);
        }
    }

    /// <summary>A fixed-capacity, time-bounded pose ring. Pinning copies only a fully covered sentence.</summary>
    public sealed class MotionPoseBuffer
    {
        private readonly MotionPoseFrame[] frames;
        private readonly double historySeconds;
        private int first;
        public int Count { get; private set; }
        public int Capacity => frames.Length;
        public double OldestTime => Count > 0 ? At(0).Time : double.NaN;
        public double NewestTime => Count > 0 ? At(Count - 1).Time : double.NaN;

        public MotionPoseBuffer(double historySeconds, int captureRate, int transformCount, int rendererCount)
        {
            if (historySeconds <= 0 || double.IsNaN(historySeconds) || double.IsInfinity(historySeconds) ||
                captureRate < 1 || historySeconds * captureRate > 100000)
                throw new ArgumentOutOfRangeException();
            this.historySeconds = historySeconds;
            frames = new MotionPoseFrame[(int)Math.Ceiling(historySeconds * captureRate) + 2];
            for (var i = 0; i < frames.Length; ++i)
                frames[i] = new MotionPoseFrame(transformCount, rendererCount);
        }

        public void Add(double time, MotionPoseFrame frame)
        {
            if (double.IsNaN(time) || double.IsInfinity(time) || (Count > 0 && time <= NewestTime))
                throw new ArgumentException("Pose timestamps must be finite and strictly increasing.");
            if (Count == Capacity)
            {
                first = (first + 1) % Capacity;
                --Count;
            }
            var target = At(Count++);
            target.CopyFrom(frame);
            target.Time = time;
            // Retain the one preceding sample required to interpolate at the 30-second boundary.
            while (Count > 2 && At(1).Time < time - historySeconds)
            {
                first = (first + 1) % Capacity;
                --Count;
            }
        }

        public bool TryPin(double start, double end, out MotionPoseClip clip)
        {
            clip = null;
            if (Count < 2 || double.IsNaN(start) || double.IsNaN(end) ||
                double.IsInfinity(start) || double.IsInfinity(end) || end <= start ||
                start < OldestTime || start < NewestTime - historySeconds || end > NewestTime)
                return false;

            var inside = 0;
            for (var i = 0; i < Count; ++i)
                if (At(i).Time > start && At(i).Time < end) ++inside;
            var pinned = new MotionPoseFrame[inside + 2];
            var shape = At(0);
            for (var i = 0; i < pinned.Length; ++i)
                pinned[i] = new MotionPoseFrame(shape.Positions.Length, shape.Visible.Length);
            SampleTime(start, pinned[0]);
            var index = 1;
            for (var i = 0; i < Count; ++i)
                if (At(i).Time > start && At(i).Time < end) pinned[index++].CopyFrom(At(i));
            SampleTime(end, pinned[index]);
            clip = new MotionPoseClip(pinned);
            return true;
        }

        public void Clear()
        {
            first = 0;
            Count = 0;
        }

        private MotionPoseFrame At(int index) => frames[(first + index) % frames.Length];

        private void SampleTime(double time, MotionPoseFrame destination)
        {
            for (var i = 1; i < Count; ++i)
            {
                var right = At(i);
                if (right.Time < time) continue;
                var left = At(i - 1);
                MotionPoseFrame.Interpolate(left, right, (time - left.Time) / (right.Time - left.Time), destination);
                destination.Time = time;
                return;
            }
            destination.CopyFrom(At(Count - 1));
        }
    }
}
