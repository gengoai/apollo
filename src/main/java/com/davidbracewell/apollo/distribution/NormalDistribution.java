package com.davidbracewell.apollo.distribution;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * The type Normal distribution.
 *
 * @author David B. Bracewell
 */
public class NormalDistribution implements RealDistribution<NormalDistribution> {
  private static final long serialVersionUID = 1L;
  private DescriptiveStatistics statistics = new DescriptiveStatistics();
  private org.apache.commons.math3.distribution.NormalDistribution wrapped = null;
  private boolean dirty = true;

  /**
   * Instantiates a new Normal distribution.
   */
  public NormalDistribution() {

  }

  /**
   * Instantiates a new Normal distribution.
   *
   * @param mean              the mean
   * @param standardDeviation the standard deviation
   */
  public NormalDistribution(double mean, double standardDeviation) {
    this.statistics = null;
    this.wrapped = new org.apache.commons.math3.distribution.NormalDistribution(mean, standardDeviation);
    this.dirty = false;
  }

  @Override
  public double probability(double value) {
    return get().probability(value);
  }

  @Override
  public double cumulativeProbability(double value) {
    return get().cumulativeProbability(value);
  }

  @Override
  public double cumulativeProbability(double lowerBound, double higherBound) {
    return get().probability(lowerBound, higherBound);
  }

  @Override
  public double inverseCumulativeProbability(double p) {
    return get().inverseCumulativeProbability(p);
  }

  @Override
  public double sample() {
    return get().sample();
  }

  private org.apache.commons.math3.distribution.NormalDistribution get() {
    if (wrapped == null || dirty) {
      dirty = false;
      wrapped = new org.apache.commons.math3.distribution.NormalDistribution(statistics.getMean(),
                                                                             statistics.getStandardDeviation());
    }
    return wrapped;
  }

  @Override
  public NormalDistribution increment(double value) {
    if (statistics == null) {
      throw new UnsupportedOperationException("Distribution initialized with a mean and standard deviation");
    }
    dirty = false;
    statistics.addValue(value);
    return this;
  }


}// END OF NormalDistribution
