package com.aircraft.analysis.Predict

final case class RankedPrediction(label: String, confidence: Double)

final case class PredictionSummary(
    label: String,
    confidence: Double,
    top3: Seq[RankedPrediction]
)

final case class ModelScore(
    accuracy: Double,
    f1: Double,
    classes: Int
)
