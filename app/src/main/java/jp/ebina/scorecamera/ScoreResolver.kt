package jp.ebina.scorecamera

/** No recognizer may silently discard a longer reading from another recognizer. */
internal object ScoreResolver {
    fun resolve(ink: InkReading, neural: Int?, neuralCount: Int, neuralConfidence: Float, ocr: Int?): Int? {
        if (ink.count == 0) return null // A blank is not a written zero.
        val detected = neural?.takeIf { neuralCount in 1..3 && neuralConfidence >= .45f }
        val text = ocr?.takeIf { it in 0..999 }
        // A noisy foreground count may not veto two agreeing whole-field readers.
        // Neither a single detector nor a partial foreground value can use this escape.
        if (!ink.reliableCount) return detected?.takeIf {
            neuralConfidence >= .55f && it == text && neuralCount == text.toString().length
        }
        if (ink.count !in 1..3) return null
        val count = maxOf(ink.count, if (detected != null) neuralCount else 0, text?.toString()?.length ?: 0)
        val completeNeural = detected?.takeIf { neuralCount == count && neuralConfidence >= .55f }
        val completeOcr = text?.takeIf { it.toString().length == count }
        val completeInk = ink.value?.takeIf { ink.count == count && ink.confidence >= .80f }
        // Agreement is only considered at the largest supported length: 30+30 beats a truncated 3.
        if (completeNeural != null && completeNeural == completeOcr) return completeNeural
        if (completeInk != null) {
            if (completeInk == completeNeural || completeInk == completeOcr) return completeInk
            // More components are not automatically more digits either (e.g. noise turns 30 into 301).
            // Any nonmatching whole-field reading needs independent corroboration, regardless of length.
            if ((detected != null && detected != completeInk) || (text != null && text != completeInk)) return null
            return completeInk.takeIf { ink.confidence >= .90f || (ink.consistentMasks && ink.confidence >= .85f) }
        }
        // The generic text OCR is a helper, not a veto on a complete, strong handwriting detection.
        // Visible glyph count still has to agree, so a high-confidence '2' cannot stand in for '12'.
        if (ink.count != count) return null // A longer, uncorroborated reading requires review, never the shorter value.
        if (completeNeural != null && neuralConfidence >= .70f) return completeNeural
        if (completeNeural != null && (completeOcr == null || completeOcr == completeNeural)) return completeNeural
        return completeOcr.takeIf { completeNeural == null }
    }

    /** A different crop/model may recover thin digits, but must agree with a different reading mechanism. */
    fun recoverFromAlternate(ink: InkReading, primaryCount: Int, primaryConfidence: Float,
        alternate: Int?, alternateCount: Int, alternateConfidence: Float, ocr: Int?): Int? {
        val value = resolve(ink, alternate, alternateCount, alternateConfidence, ocr) ?: return null
        val minimumCount = maxOf(if (primaryConfidence >= .45f) primaryCount else 0,
            if (ink.reliableCount) ink.count else 0)
        if (alternateCount < minimumCount || alternate != value || alternateConfidence < .45f) return null
        return value.takeIf { (ink.reliableCount && ink.value == value && ink.confidence >= .80f) || ocr == value }
    }

    fun recoverFromPreprocessing(original: InkReading, alternative: InkReading, observedCount: Int,
        neural: Int?, count: Int, confidence: Float, ocr: Int?): Int? {
        // Changing preprocessing must not vote away a reliably observed extra glyph.
        if (original.reliableCount && original.count > alternative.count) return null
        return recoverFromAlternate(alternative, observedCount, if (observedCount > 0) 1f else 0f,
            neural, count, confidence, ocr)
    }
}
