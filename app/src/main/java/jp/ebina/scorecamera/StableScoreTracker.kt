package jp.ebina.scorecamera

/** Confirmation requires two complete observations, never a mixture of fields from different sheets. */
internal class StableScoreTracker {
    private var previous: List<Int?>? = null
    private var lastSeen = 0L
    private var matches = 0
    @Synchronized fun observe(scores: List<Int?>, now: Long): Boolean {
        if (scores.isEmpty() || scores.any { it == null }) { reset(); return false }
        matches = if (previous == scores && now - lastSeen <= 30_000) matches + 1 else 1
        previous = scores.toList(); lastSeen = now
        return matches >= 2
    }
    @Synchronized fun reset() { previous = null; matches = 0; lastSeen = 0 }
}
