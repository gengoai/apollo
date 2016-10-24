package com.davidbracewell.apollo.distribution;

import com.davidbracewell.Copyable;
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
      this.nSuccess = numberOfSuccess;
      this.trials = numberOfTrials;
      this.randomGenerator = randomGenerator;
   }

   @Override
   public Binomial copy() {
      return new Binomial(nSuccess, trials);
   }


   @Override
   public double probability(double x) {
      return getDistribution().probability((int) x);
   }


   @Override
   public double getMode() {
      return Math.floor((trials + 1) * (nSuccess / trials));
   }

   @Override
   public double getMean() {
      return nSuccess * trials;
   }

   @Override
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
    * Get number of successes int.
    *
    * @return the int
    */
   public int getNumberOfSuccesses() {
      return nSuccess;
   }

   /**
    * Gets number of failures.
    *
    * @return the number of failures
    */
   public int getNumberOfFailures() {
      return trials - nSuccess;
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
   public double cumulativeProbability(double x) {
      return getDistribution().cumulativeProbability((int) x);
   }

   @Override
   public double cumulativeProbability(double lowerBound, double higherBound) {
      return getDistribution().cumulativeProbability((int) lowerBound, (int) higherBound);
   }

   @Override
   public double inverseCumulativeProbability(double p) {
      return getDistribution().inverseCumulativeProbability(p);
   }

   @Override
   public int sample() {
      return getDistribution().sample();
   }

   @Override
   public Binomial increment(int variable, int numberOfObservations) {
      if (numberOfObservations != 0) {
         if (variable > 0) {
            nSuccess += numberOfObservations;
         }
         trials += numberOfObservations;
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
               BinomialDistribution binomial = new BinomialDistribution(randomGenerator, trials,
                                                                        probabilityOfSuccess());
               wrapped = binomial;
               return binomial;
            }
         }
      }
      return wrapped;
   }

}// END OF Binomial
