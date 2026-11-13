package jp.ebina.scorecamera

import org.junit.Assert.*
import org.junit.Test

class InkConsensusTest {
    private fun reading(value: Int?, count: Int) = InkReading(value, count, .95f, true)

    @Test fun twoCompleteMasksRecoverADigitLostByAnotherMask() {
        val result = InkConsensus.combine(listOf(reading(4, 1), reading(42, 2), reading(42, 2)))
        assertEquals(42, result.value)
        assertTrue(result.reliableCount)
        assertTrue(result.consistentMasks)
    }
    @Test fun majorityCannotVoteAwayAnAdditionalVisibleDigit() {
        val result = InkConsensus.combine(listOf(reading(2, 1), reading(2, 1), reading(12, 2)))
        assertNull(result.value)
        assertFalse(result.reliableCount)
    }
    @Test fun conflictingCompleteMasksAreNotConfirmed() {
        assertNull(InkConsensus.combine(listOf(reading(12, 2), reading(17, 2), reading(12, 2))).value)
    }
    @Test fun emptyMasksRemainEmptyRatherThanZero() {
        val result = InkConsensus.combine(List(3) { reading(null, 0) })
        assertEquals(0, result.count)
        assertNull(result.value)
    }
}
