package jp.ebina.scorecamera

/** A majority may corroborate the largest visible count, never vote away an extra digit. */
internal object InkConsensus {
    fun combine(readings: List<InkReading>): InkReading {
        val count = readings.maxOfOrNull { it.count } ?: 0
        val complete = readings.filter { it.reliableCount && it.count == count }
        if (complete.size < 2) return InkReading(null, count, 0f, false)
        val recognized = complete.filter { it.value != null }
        if (recognized.map { it.value }.distinct().size > 1) return InkReading(null, count, 0f, true)
        return recognized.maxByOrNull { it.confidence }?.copy(consistentMasks = recognized.size >= 2)
            ?: InkReading(null, count, 0f, true)
    }
}
