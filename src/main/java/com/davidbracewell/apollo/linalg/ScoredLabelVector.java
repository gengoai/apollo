package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.analysis.Optimum;
import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.util.Comparator;

/**
 * <p>Associates a score and a label with a vector</p>
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = true)
public class ScoredLabelVector extends LabeledVector implements Comparable<ScoredLabelVector> {
   private static final long serialVersionUID = 1L;
   private final double score;

   /**
    * Instantiates a new Scored Labeled vector.
    *
    * @param label    the label
    * @param delegate the vector to wrap
    * @param score    the score
    */
   public ScoredLabelVector(Object label, @NonNull Vector delegate, double score) {
      super(label, delegate);
      this.score = score;
   }

   /**
    * Instantiates a new Scored label vector.
    *
    * @param labeledVector the labeled vector
    * @param score         the score
    */
   public ScoredLabelVector(@NonNull LabeledVector labeledVector, double score) {
      super(labeledVector.getLabel(), labeledVector);
      this.score = score;
   }

   public static Comparator<ScoredLabelVector> comparator(@NonNull Optimum optimum) {
      return (v1, v2) -> optimum.compare(v1.getScore(), v2.getScore());
   }

   /**
    * Gets the score of the vector.
    *
    * @return the score
    */
   public double getScore() {
      return score;
   }

   @Override
   public int compareTo(@NonNull ScoredLabelVector o) {
      return Double.compare(score, o.score);
   }

   @Override
   public String toString() {
      return "(" + getLabel() + ", " + score + ')';
   }

}// END OF ScoredLabelVector
