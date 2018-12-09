package com.gengoai.apollo.stat.distribution;

import com.gengoai.Copyable;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

/**
 * <p>The binomial distribution is a discrete distribution modeling the number of successes and failures.</p>
 *
 * @author David B. Bracewell
 */
public final class Binomial implements UnivariateDiscreteDistribution<Binomial>, Copyable<Binomial> {
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
   public Binomial(int numberOfSuccess, int numberOfTrials, RandomGenerator randomGenerator) {
      this.nSuccess = numberOfSuccess;
      this.trials = numberOfTrials;
      this.randomGenerator = randomGenerator;
   }

   @Override
   public Binomial copy() {
      return new Binomial(nSuccess, trials);
   }

   @Override
   public double cumulativeProbability(double x) {
      return getDistribution().cumulativeProbability((int) x);
   }

   @Override
   public double cumulativeProbability(double lowerBound, double higherBound) {
      return getDistribution().cumulativeProbability((int) lowerBound, (int) higherBound);
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

   @Override
   public double getMean() {
      return nSuccess;
   }

   @Override
   public double getMode() {
      return Math.floor((trials + 1) * (nSuccess / trials));
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
    * Get number of successes int.
    *
    * @return the int
    */
   public int getNumberOfSuccesses() {
      return nSuccess;
   }

   /**
    * Gets the number of trials.
    *
    * @return the number of trials
    */
   public int getNumberOfTrials() {
      return trials;
   }

   @Override
   public double getVariance() {
      return nSuccess * (1.0 - probabilityOfSuccess());
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

   @Override
   public double inverseCumulativeProbability(double p) {
      return getDistribution().inverseCumulativeProbability(p);
   }

   @Override
   public double probability(double x) {
      return getDistribution().probability((int) x);
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
   public int sample() {
      return getDistribution().sample();
   }

}// END OF Binomial
