package jp.ebina.scorecamera

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StableScoreTrackerTest {
    @Test fun changedOrIncompleteSheetsDoNotReuseThePreviousConfirmation() {
        val tracker = StableScoreTracker()
        assertFalse(tracker.observe(listOf(6, 13, 42, 10), 1000))
        assertTrue(tracker.observe(listOf(6, 13, 42, 10), 4000))
        assertFalse(tracker.observe(listOf(6, 13, 42, 11), 7000))
        assertFalse(tracker.observe(listOf(6, null, 42, 11), 10000))
        assertFalse(tracker.observe(listOf(6, 13, 42, 11), 13000))
    }
    @Test fun longGapsAndRefocusingRequireFreshConfirmation() {
        val tracker = StableScoreTracker()
        assertFalse(tracker.observe(listOf(12, 95, 84), 1000))
        assertFalse(tracker.observe(listOf(12, 95, 84), 34000))
        assertTrue(tracker.observe(listOf(12, 95, 84), 37000))
        tracker.reset()
        assertFalse(tracker.observe(listOf(12, 95, 84), 40000))
    }
}
