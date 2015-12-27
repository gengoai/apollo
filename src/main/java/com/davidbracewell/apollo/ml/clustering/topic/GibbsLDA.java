package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.ml.clustering.Clustering;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class GibbsLDA extends Clusterer {
  private int K;
  private double alpha;
  private double beta;

  @Override
  public Clustering cluster(List<LabeledVector> instances) {
    return null;
  }


  public int getK() {
    return K;
  }

  public void setK(int k) {
    K = k;
  }

  public double getAlpha() {
    return alpha;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  public double getBeta() {
    return beta;
  }

  public void setBeta(double beta) {
    this.beta = beta;
  }
}// END OF GibbsLDA
