package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.classification.Classification;

import java.io.Serializable;

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
   * Get labels string [ ].
   *
   * @return the string [ ]
   */
  public String[] getLabels() {
    return labels;
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
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return labels.length;
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

}// END OF Labeling
