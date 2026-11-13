package jp.ebina.scorecamera

/** One paper at a time. A reset invalidates work already running on the camera executor. */
internal class LiveScoreSession {
    @Volatile var generation: Int = 0
        private set
    private val tracker = StableScoreTracker()
    private var exportableScores: List<Int?>? = null
    @Volatile private var nextAttemptAt = 0L
    @Volatile var isHeld = false
        private set
    private var provisional = false

    data class Reading(val scores: List<Int?>, val total: Int?, val confirmed: Boolean)

    @Synchronized fun reset() {
        generation++
        tracker.reset()
        exportableScores = null
        nextAttemptAt = 0L
        isHeld = false
        provisional = false
    }

    fun coolingDown(now: Long): Boolean = now < nextAttemptAt

    @Synchronized fun completed(token: Int, now: Long) {
        if (token == generation) nextAttemptAt = now + if (provisional) 600L else 1800L
    }

    @Synchronized fun accept(token: Int, scores: List<Int?>, now: Long): Reading? {
        if (token != generation) return null
        if (isHeld) return null
        val snapshot = scores.toList()
        val total = if (snapshot.isNotEmpty() && snapshot.all { it != null }) snapshot.filterNotNull().sum() else null
        val confirmed = tracker.observe(snapshot, now)
        provisional = total != null && !confirmed
        // Delivery can reach the UI after the executor has already recorded completion.
        if (provisional) nextAttemptAt = minOf(nextAttemptAt, now + 600L)
        isHeld = confirmed
        // Never send a provisional total to the ordinary (confirmed) result screen.
        exportableScores = if (confirmed || total == null) snapshot else null
        return Reading(snapshot, total, confirmed)
    }

    @Synchronized fun failed(token: Int): Boolean {
        if (token != generation) return false
        if (isHeld) return false
        tracker.reset()
        exportableScores = null
        provisional = false
        return true
    }

    @Synchronized fun resultScores(): IntArray? = exportableScores?.map { it ?: -1 }?.toIntArray()
}
