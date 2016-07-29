package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;

import java.util.HashMap;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class TopicCluster extends Cluster {
  private static final long serialVersionUID = 1L;
  private final Map<Vector, Double> scores = new HashMap<>();

  public void addPoint(Vector vector, double score) {
    super.addPoint(vector);
    scores.put(vector, score);
  }

  @Override
  public double getScore(Vector vector) {
    return scores.getOrDefault(vector, 0.0);
  }

  @Override
  public void clear() {
    super.clear();
    scores.clear();
  }

}// END OF TopicCluster
