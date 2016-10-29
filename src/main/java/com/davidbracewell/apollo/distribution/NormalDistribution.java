package com.davidbracewell.apollo.distribution;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * The type Normal distribution.
 *
 * @author David B. Bracewell
 */
public class NormalDistribution implements UnivariateRealDistribution<NormalDistribution> {
   private static final long serialVersionUID = 1L;
   private DescriptiveStatistics statistics = new DescriptiveStatistics();
   private org.apache.commons.math3.distribution.NormalDistribution wrapped = null;

   /**
    * Instantiates a new Normal distribution with mean and standard deviation 0.
    */
   public NormalDistribution() {
      this.wrapped = new org.apache.commons.math3.distribution.NormalDistribution(0, 1e-25);
   }

   /**
    * Instantiates a new Normal distribution.
    *
    * @param mean              the mean
    * @param standardDeviation the standard deviation
    */
   public NormalDistribution(double mean, double standardDeviation) {
      this.wrapped = new org.apache.commons.math3.distribution.NormalDistribution(mean, standardDeviation);
      for (double v : this.wrapped.sample(100)) {
         this.statistics.addValue(v);
      }
   }

   @Override
   public double getMode() {
      return statistics.getMean();
   }

   @Override
   public double getMean() {
      return statistics.getMean();
   }

   @Override
   public double getVariance() {
      return statistics.getVariance();
   }

   @Override
   public double probability(double x) {
      return getDistribution().density(x);
   }

   @Override
   public double cumulativeProbability(double value) {
      return getDistribution().cumulativeProbability(value);
   }

   @Override
   public double cumulativeProbability(double lowerBound, double higherBound) {
      return getDistribution().probability(lowerBound, higherBound);
   }

   @Override
   public double inverseCumulativeProbability(double p) {
      return getDistribution().inverseCumulativeProbability(p);
   }

   @Override
   public double sample() {
      return getDistribution().sample();
   }

   private org.apache.commons.math3.distribution.NormalDistribution getDistribution() {
      if (wrapped == null) {
         synchronized (this) {
            if (wrapped == null) {
               double mean = statistics.getN() > 0 ? statistics.getMean() : 0;
               double std = statistics.getN() > 0 ? statistics.getStandardDeviation() : 0;
               org.apache.commons.math3.distribution.NormalDistribution n =
                  new org.apache.commons.math3.distribution.NormalDistribution(mean, std);
               wrapped = n;
               return n;
            }
         }
      }
      return wrapped;
   }

   @Override
   public NormalDistribution addValue(double value) {
      if (statistics == null) {
         throw new UnsupportedOperationException("Distribution initialized with a mean and standard deviation");
      }
      this.wrapped = null;
      statistics.addValue(value);
      return this;
   }


}// END OF NormalDistribution

