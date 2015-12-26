package com.davidbracewell.apollo.distribution;

import com.davidbracewell.apollo.ApolloMath;

/**
 * @author David B. Bracewell
 */
public class Binomial implements DiscreteDistribution<Binomial> {
  private final double probabilityOfSuccess;
  private int nSuccess = 0;
  private int trials = 0;

  public Binomial(double probabilityOfSuccess) {
    this.probabilityOfSuccess = probabilityOfSuccess;
  }

  public static void main(String[] args) {
    Binomial binomial = new Binomial(0.5);
    binomial.increment(1, 100);
    binomial.increment(0, 100);
    System.out.println(binomial.probabilityOfSuccess());
    System.out.println(binomial.probability(100));
  }

  @Override
  public double probability(int value) {
    return Math.exp(logProbability(value));
  }

  public double probabilityOfSuccess() {
    return probabilityOfSuccess;
  }

  @Override
  public double logProbability(int value) {
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
    return this;
  }

}// END OF Binomial
