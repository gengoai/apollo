package com.davidbracewell.apollo.distribution;

import com.davidbracewell.Copyable;
import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

/**
 * <p>The binomial distribution is a discrete distribution modeling the number of successes and failures.</p>
 *
 * @author David B. Bracewell
 */
public final class Binomial implements DiscreteDistribution<Binomial>, Copyable<Binomial> {
   private static final long serialVersionUID = 1L;
   private final RandomGenerator randomGenerator;
   private int nSuccess = 0;
   private int trials = 0;
   private volatile BinomialDistribution wrapped = null;

   /**
    * Instantiates a new Binomial.
    */
   public Binomial() {
      this(0, 0);
   }

   /**
    * Instantiates a new Binomial.
    *
    * @param numberOfSuccess the number of successes
    * @param numberOfTrials  the number of trials
    */
   public Binomial(int numberOfSuccess, int numberOfTrials) {
      this(numberOfSuccess, numberOfTrials, new Well19937c());

   }

   /**
    * Instantiates a new Binomial.
    *
    * @param numberOfSuccess the number of successes
    * @param numberOfTrials  the number of trials
    * @param randomGenerator the random generator to use for sampling
    */
   public Binomial(int numberOfSuccess, int numberOfTrials, @NonNull RandomGenerator randomGenerator) {
      Preconditions.checkArgument(numberOfTrials > 0, "Number of trails must be > 0");
      this.nSuccess = numberOfSuccess;
      this.trials = numberOfTrials;
      this.randomGenerator = randomGenerator;
   }

   @Override
   public Binomial copy() {
      return new Binomial(nSuccess, trials);
   }


   @Override
   public double probability(int value) {
      return getDistribution().probability(value);
   }

   /**
    * Gets mean.
    *
    * @return the mean
    */
   public double getMean() {
      return nSuccess * trials;
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
    * Gets the number of trials.
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
      return getDistribution().logProbability(value);
   }

   @Override
   public double cumulativeProbability(int x) {
      return getDistribution().cumulativeProbability(x);
   }

   @Override
   public double cumulativeProbability(int lowerBound, int higherBound) {
      return getDistribution().cumulativeProbability(lowerBound, higherBound);
   }

   @Override
   public int sample() {
      return getDistribution().sample();
   }

   @Override
   public Binomial increment(int k, long value) {
      if (value > 0) {
         if (k > 0) {
            nSuccess += value;
         }
         trials += value;
         if (trials < 0) {
            trials = 0;
            nSuccess = 0;
         }
         this.wrapped = null;
      }
      return this;
   }

   private BinomialDistribution getDistribution() {
      if (wrapped == null) {
         synchronized (this) {
            if (wrapped == null) {
               wrapped = new BinomialDistribution(randomGenerator, trials, probabilityOfSuccess());
            }
         }
      }
      return wrapped;
   }

}// END OF Binomial
