package com.davidbracewell.apollo.ml.sequence.decoder;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DecoderRow implements Serializable, Comparable<DecoderRow> {
  private final List<String> labels = new ArrayList<>();
  private final List<Double> probs = new ArrayList<>();
  private double rowProb = 0;

  public DecoderRow() {

  }

  public DecoderRow(DecoderRow copy) {
    this.labels.addAll(copy.labels);
    this.probs.addAll(copy.probs);
    this.rowProb = copy.rowProb;
  }

  public DecoderRow(DecoderRow copy, String label, double p) {
    this(copy);
    add(label, p);
  }

  @Override
  public int compareTo(DecoderRow o) {
    return Double.compare(rowProb, o.rowProb);
  }

  public void add(String label, double prob) {
    labels.add(label);
    probs.add(prob);
    rowProb += Math.log(prob);
  }

  public String getLabel(int i) {
    return labels.get(i);
  }

  public double getProbability(int i) {
    return probs.get(i);
  }

  public double getRowProbability() {
    return rowProb;
  }

  public int size() {
    return labels.size();
  }

}// END OF DecoderRow
