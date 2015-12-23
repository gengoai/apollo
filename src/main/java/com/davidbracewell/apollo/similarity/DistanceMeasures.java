package com.davidbracewell.apollo.similarity;


/**
 * The enum Distance measures.
 *
 * @author dbracewell
 */
public enum DistanceMeasures {
  /**
   * The EUCLIDEAN.
   */
  EUCLIDEAN(new EuclideanDistance()),
  /**
   * The MANHATTAN.
   */
  MANHATTAN(new ManhattanDistance()),
  /**
   * The HAMMING.
   */
  HAMMING(new HammingDistance()),
  /**
   * The COSINE.
   */
  COSINE(new OneMinusSimilarityDistance(new CosineSimilarity()));


  /**
   * The Distance measure.
   */
  final DistanceMeasure distanceMeasure;

  /**
   * Instantiates a new Distance measures.
   *
   * @param distanceMeasure the distance measure
   */
  DistanceMeasures(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

  /**
   * Get distance measure.
   *
   * @return the distance measure
   */
  public DistanceMeasure get() {
    return this.distanceMeasure;
  }

}//END OF DistanceMeasures
