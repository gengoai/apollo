package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.affinity.Optimum;
import lombok.NonNull;

import java.util.Comparator;

/**
 * The type Scored label vector.
 *
 * @author David B. Bracewell
 */
public class ScoredLabelVector extends LabeledVector implements Comparable<ScoredLabelVector> {
  private static final long serialVersionUID = 1L;
  /**
   * The Score.
   */
  final double score;

  /**
   * Instantiates a new Labeled vector.
   *
   * @param label    the label
   * @param delegate the delegate
   * @param score    the score
   */
  public ScoredLabelVector(Object label, Vector delegate, double score) {
    super(label, delegate);
    this.score = score;
  }

  /**
   * Instantiates a new Scored label vector.
   *
   * @param labeledVector the labeled vector
   * @param score         the score
   */
  public ScoredLabelVector(LabeledVector labeledVector, double score) {
    super(labeledVector.getLabel(), labeledVector);
    this.score = score;
  }

  public static Comparator<ScoredLabelVector> comparator(@NonNull Optimum optimum) {
    return (v1, v2) -> optimum.compare(v1.getScore(), v2.getScore());
  }

  /**
   * Gets score.
   *
   * @return the score
   */
  public double getScore() {
    return score;
  }

  @Override
  public int compareTo(ScoredLabelVector o) {
    if (o == null) return -1;
    return Double.compare(score, o.score);
  }

  @Override
  public String toString() {
    return "(" + getLabel() + ", " + score + ')';
  }
}// END OF ScoredLabelVector
