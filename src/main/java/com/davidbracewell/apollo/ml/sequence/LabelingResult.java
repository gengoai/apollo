package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.classification.ClassifierResult;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class LabelingResult implements Serializable {
  private static final long serialVersionUID = 1L;
  private final String[] labels;
  private final double[] probs;
  private double sequenceProbability;

  public LabelingResult(int size) {
    this.labels = new String[size];
    this.probs = new double[size];
  }

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
      return "****START****";
    } else if (index >= labels.length) {
      return "****END****";
    }
    return labels[index];
  }

  public int size() {
    return labels.length;
  }

  public double getProbability(int index) {
    return probs[index];
  }

  public void setLabel(int index, String label, double probability) {
    labels[index] = label;
    probs[index] = probability;
  }

  public void setLabel(int index, ClassifierResult result) {
    labels[index] = result.getResult();
    probs[index] = result.getConfidence();
  }

  public double getSequenceProbability() {
    return sequenceProbability;
  }

  public void setSequenceProbability(double sequenceProbability) {
    this.sequenceProbability = sequenceProbability;
  }

}// END OF LabelingResult
