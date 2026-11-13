package jp.ebina.scorecamera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ScoreResolverTest {
    @Test fun aMissingTensDigitMustNotBecomeASingleDigitScore() {
        assertNull(ScoreResolver.resolve(InkReading(null, 2, .5f, true), 2, 1, .9f, 2))
    }
    @Test fun aLongerForegroundNeedsCorroborationWhenAnotherReaderDisagrees() {
        assertNull(ScoreResolver.resolve(InkReading(12, 2, .99f, true), 2, 1, .9f, null))
    }
    @Test fun disagreementBetweenCompleteReadingsRequiresConfirmation() {
        assertNull(ScoreResolver.resolve(InkReading(13, 2, .99f, true), 12, 2, .9f, null))
    }
    @Test fun anEmptyBoxIsNotAZero() {
        assertNull(ScoreResolver.resolve(InkReading(null, 0, 0f, true), 0, 1, .9f, 0))
    }
    @Test fun aWrittenZeroRemainsARealZero() {
        assertEquals(0, ScoreResolver.resolve(InkReading(0, 1, .99f, true), 0, 1, .9f, 0))
    }
    @Test fun unstableSegmentationCannotConfirmAPartialNumber() {
        assertNull(ScoreResolver.resolve(InkReading(12, 2, .99f, false), 12, 2, .9f, null))
    }
    @Test fun genericTextOcrDoesNotVetoStrongCompleteHandwriting() {
        assertEquals(84, ScoreResolver.resolve(InkReading(null, 2, .5f, true), 84, 2, .85f, 88))
    }
    @Test fun matchingCompleteForegroundReadingsCanRecoverAPartialDetector() {
        assertEquals(12, ScoreResolver.resolve(InkReading(12, 2, .87f, true, true), 2, 1, .9f, 12))
    }
    @Test fun aConfidentPartialForegroundCannotVetoTwoCompleteReaders() {
        assertEquals(30, ScoreResolver.resolve(InkReading(3, 1, .999f, true, true), 30, 2, .9f, 30))
        assertEquals(20, ScoreResolver.resolve(InkReading(0, 1, .999f, true, true), 20, 2, .9f, 20))
    }
    @Test fun aSingleLongerDetectorStillVetoesTheShorterForeground() {
        assertNull(ScoreResolver.resolve(InkReading(2, 1, .999f, true, true), 32, 2, .9f, null))
        assertNull(ScoreResolver.resolve(InkReading(1, 1, .999f, true, true), 31, 2, .5f, null))
    }
    @Test fun agreeingWholeFieldReadersCanRecoverNoisySegmentation() {
        assertEquals(24, ScoreResolver.resolve(InkReading(null, 3, 0f, false), 24, 2, .9f, 24))
    }
    @Test fun longerInkCannotBeVotedAwayByTwoPartialReaders() {
        assertNull(ScoreResolver.resolve(InkReading(null, 3, 0f, true), 12, 2, .9f, 12))
    }
    @Test fun extraForegroundComponentsCannotAppendADigitToACompleteReading() {
        assertNull(ScoreResolver.resolve(InkReading(301, 3, .999f, true, true), 30, 2, .9f, null))
        assertNull(ScoreResolver.resolve(InkReading(130, 3, .999f, true, true), 30, 2, .9f, null))
    }
    @Test fun aWeakMatchingDetectorIsNotAContradiction() {
        assertEquals(12, ScoreResolver.resolve(InkReading(12, 2, .99f, true, true), 12, 2, .48f, null))
    }
    @Test fun alternateCropNeedsIndependentAgreementAndMustNotLoseDigits() {
        assertEquals(18, ScoreResolver.recoverFromAlternate(InkReading(18, 2, .99f, true, true), 1, .9f, 18, 2, .8f, null))
        assertEquals(20, ScoreResolver.recoverFromAlternate(InkReading(null, 2, 0f, false), 2, .9f, 20, 2, .8f, 20))
        assertNull(ScoreResolver.recoverFromAlternate(InkReading(null, 2, 0f, true), 2, .9f, 29, 2, .9f, 20))
        assertNull(ScoreResolver.recoverFromAlternate(InkReading(2, 1, .99f, true, true), 2, .9f, 2, 1, .9f, 2))
    }
    @Test fun grayscaleRecoveryNeedsIndependentAgreementWithoutLosingAGlyph() {
        assertEquals(12, ScoreResolver.recoverFromPreprocessing(InkReading(17,2,.99f,true),
            InkReading(12,2,.99f,true),2,12,2,.9f,null))
        assertNull(ScoreResolver.recoverFromPreprocessing(InkReading(null,3,0f,true),
            InkReading(12,2,.99f,true),2,12,2,.9f,12))
        assertNull(ScoreResolver.recoverFromPreprocessing(InkReading(null,2,0f,false),
            InkReading(2,1,.99f,true),2,2,1,.9f,2))
        assertEquals(24, ScoreResolver.recoverFromPreprocessing(InkReading(null,3,0f,false),
            InkReading(24,2,.99f,true),2,24,2,.9f,null))
    }
}
