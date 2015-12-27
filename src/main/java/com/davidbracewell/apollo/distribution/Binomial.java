package com.davidbracewell.apollo.distribution;

import com.davidbracewell.apollo.ApolloMath;

/**
 * The type Binomial.
 *
 * @author David B. Bracewell
 */
public class Binomial implements DiscreteDistribution<Binomial> {
  private int nSuccess = 0;
  private int trials = 0;

  /**
   * Instantiates a new Binomial.
   */
  public Binomial() {
    this(0, 0);
  }

  /**
   * Instantiates a new Binomial.
   *
   * @param numberOfSuccess the number of success
   * @param numberOfTrials  the number of trials
   */
  public Binomial(int numberOfSuccess, int numberOfTrials) {
    this.nSuccess = numberOfSuccess;
    this.trials = numberOfTrials;
  }

  @Override
  public double probability(int value) {
    double logP = logProbability(value);
    return Double.isFinite(logP) ? Math.exp(logP) : 0d;
  }

  /**
   * Gets mean.
   *
   * @return the mean
   */
  public double getMean() {
    return nSuccess;
  }

  /**
   * Gets variance.
   *
   * @return the variance
   */
  public double getVariance() {
    return nSuccess * (1.0 - probabilityOfSuccess());
  }


  /**
   * Gets number of trials.
   *
   * @return the number of trials
   */
  public int getNumberOfTrials() {
    return trials;
  }

  /**
   * Probability of success double.
   *
   * @return the double
   */
  public double probabilityOfSuccess() {
    return (double) nSuccess / (double) trials;
  }

  @Override
  public double logProbability(int value) {
    if (trials <= 0 || value < 0 || value > trials) {
      return Double.NEGATIVE_INFINITY;
    }
    double probabilityOfSuccess = probabilityOfSuccess();
    return ApolloMath.logGamma(trials + 1) -
      ApolloMath.logGamma(value + 1) -
      ApolloMath.logGamma(trials - value + 1) +
      value * Math.log(probabilityOfSuccess) + (trials - value) * Math.log(1 - probabilityOfSuccess);
  }

  @Override
  public int sample() {
    return 0;
  }

  @Override
  public Binomial increment(int k, long value) {
    if (k > 0) {
      nSuccess += value;
    }
    trials += value;
    if (trials < 0) {
      trials = 0;
      nSuccess = 0;
    }
    return this;
  }

}// END OF Binomial
