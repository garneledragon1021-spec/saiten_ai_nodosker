package jp.ebina.scorecamera

import org.junit.Assert.*
import org.junit.Test

class LiveScoreSessionTest {
    @Test fun nextPaperSkipsOldCooldownIncludingLateCompletions() {
        val session = LiveScoreSession()
        session.completed(0, 100)
        assertTrue(session.coolingDown(101))
        session.reset()
        assertFalse(session.coolingDown(102))
        session.completed(0, 200)
        assertFalse(session.coolingDown(201))
        session.completed(1, 300)
        assertTrue(session.coolingDown(301))
        assertFalse(session.coolingDown(2100))
    }
    @Test fun firstReadingShowsProvisionalTotalButDoesNotExportItAsConfirmed() {
        val session = LiveScoreSession()
        val first = session.accept(0, listOf(6, 13, 42, 10), 100)!!
        assertEquals(71, first.total)
        assertFalse(first.confirmed)
        assertNull(session.resultScores())
        assertTrue(session.accept(0, first.scores, 200)!!.confirmed)
        assertArrayEquals(intArrayOf(6, 13, 42, 10), session.resultScores())
    }

    @Test fun nextPaperClearsResultsRejectsOldWorkAndRequiresTwoFreshReadings() {
        val session = LiveScoreSession()
        val old = session.generation
        val scores = listOf(10, 14, 23, 24)
        session.accept(old, scores, 100)
        session.accept(old, scores, 200)
        session.reset()
        assertNull(session.resultScores())
        assertNull(session.accept(old, scores, 300))
        assertFalse(session.accept(session.generation, scores, 400)!!.confirmed)
        assertFalse(session.failed(old))
        assertTrue(session.accept(session.generation, scores, 500)!!.confirmed)
    }

    @Test fun changedReadingReplacesTheProvisionalTotalAndRestartsConfirmation() {
        val session = LiveScoreSession()
        session.accept(0, listOf(6, 13, 42, 10), 100)
        val next = session.accept(0, listOf(8, 20, 27, 18), 200)!!
        assertEquals(73, next.total)
        assertFalse(next.confirmed)
        assertNull(session.resultScores())
    }

    @Test fun unreadFieldIsNotZeroButActualZeroIsAllowed() {
        val session = LiveScoreSession()
        assertNull(session.accept(0, listOf(6, null, 42, 10), 100)!!.total)
        assertArrayEquals(intArrayOf(6, -1, 42, 10), session.resultScores())
        assertEquals(0, session.accept(0, listOf(0, 0, 0, 0), 200)!!.total)
        assertNull(session.resultScores())
    }

    @Test fun currentFailureClearsPreviousConfirmation() {
        val session = LiveScoreSession()
        val scores = listOf(6, 13, 42, 10)
        session.accept(0, scores, 100)
        assertTrue(session.failed(0))
        assertNull(session.resultScores())
        assertFalse(session.accept(0, scores, 300)!!.confirmed)
    }

    @Test fun confirmedResultIsHeldUntilExplicitReset() {
        val session = LiveScoreSession()
        val scores = listOf(6, 13, 42, 10)
        session.accept(0, scores, 100); session.accept(0, scores, 200)
        assertTrue(session.isHeld)
        assertNull(session.accept(0, listOf(0, 0, 0, 0), 300))
        assertFalse(session.failed(0))
        assertArrayEquals(intArrayOf(6, 13, 42, 10), session.resultScores())
        session.reset()
        assertFalse(session.isHeld)
        assertNull(session.resultScores())
    }

    @Test fun secondObservationUsesShortCooldownRegardlessOfUiDeliveryOrder() {
        val beforeCompletion = LiveScoreSession()
        beforeCompletion.accept(0, listOf(1, 2), 100)
        beforeCompletion.completed(0, 101)
        assertTrue(beforeCompletion.coolingDown(700))
        assertFalse(beforeCompletion.coolingDown(701))
        val afterCompletion = LiveScoreSession()
        afterCompletion.completed(0, 100)
        afterCompletion.accept(0, listOf(1, 2), 101)
        assertFalse(afterCompletion.coolingDown(701))
        afterCompletion.failed(0)
        afterCompletion.completed(0, 800)
        assertTrue(afterCompletion.coolingDown(2599))
        assertFalse(afterCompletion.coolingDown(2600))
    }
}
