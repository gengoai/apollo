package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.classification.Classification;
import com.gengoai.apollo.ml.Instance;

import java.io.Serializable;
import java.util.Optional;

/**
 * The type Labeling.
 *
 * @author David B. Bracewell
 */
public class Labeling implements Serializable {
   private static final long serialVersionUID = 1L;
   private final String[] labels;
   private final double[] probs;
   private double sequenceProbability;


   /**
    * Instantiates a new Labeling.
    *
    * @param size the size
    */
   public Labeling(int size) {
      this.labels = new String[size];
      this.probs = new double[size];
   }

   /**
    * Gets label.
    *
    * @param index the index
    * @return the label
    */
   public String getLabel(int index) {
      if (index < 0) {
         return Sequence.BOS;
      } else if (index >= labels.length) {
         return Sequence.EOS;
      }
      return labels[index];
   }

   /**
    * Get labels string [ ].
    *
    * @return the string [ ]
    */
   public String[] getLabels() {
      return labels;
   }

   /**
    * Gets probability.
    *
    * @param index the index
    * @return the probability
    */
   public double getProbability(int index) {
      return probs[index];
   }

   /**
    * Gets sequence probability.
    *
    * @return the sequence probability
    */
   public double getSequenceProbability() {
      return sequenceProbability;
   }

   /**
    * Sets sequence probability.
    *
    * @param sequenceProbability the sequence probability
    */
   public void setSequenceProbability(double sequenceProbability) {
      this.sequenceProbability = sequenceProbability;
   }

   public Context<Instance> iterator(Sequence sequence) {
      return new LabelingContext(sequence);
   }

   public Context<Instance> iterator(Sequence sequence, int index) {
      return new LabelingContext(sequence, index);
   }

   private class LabelingContext extends Context<Instance> {
      private final Sequence sequence;


      public LabelingContext(Sequence sequence) {
         this.sequence = sequence;
      }

      public LabelingContext(Sequence sequence, int index) {
         this.sequence = sequence;
         setIndex(index);
      }

      @Override
      protected Optional<Instance> getContextAt(int index) {
         return index >= 0 && index < sequence.size() ? Optional.of(sequence.get(index)) : Optional.empty();
      }

      @Override
      protected Optional<String> getLabelAt(int index) {
         return index >= 0 && index < labels.length ? Optional.of(labels[index]) : Optional.empty();
      }

      @Override
      public int size() {
         return labels.length;
      }
   }

   /**
    * Sets label.
    *
    * @param index  the index
    * @param result the result
    */
   public void setLabel(int index, Classification result) {
      labels[index] = result.getResult();
      probs[index] = result.getConfidence();
   }

   /**
    * Sets label.
    *
    * @param index       the index
    * @param label       the label
    * @param probability the probability
    */
   public void setLabel(int index, String label, double probability) {
      labels[index] = label;
      probs[index] = probability;
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return labels.length;
   }

}// END OF Labeling
