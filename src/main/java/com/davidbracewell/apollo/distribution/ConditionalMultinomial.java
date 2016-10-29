package com.davidbracewell.apollo.distribution;

import com.davidbracewell.Copyable;

import java.util.Arrays;

/**
 * <p>
 * A distribution representing a discrete event conditioned on another discrete event, i.e. <code>P(E|X=x)</code>.
 * </p>
 *
 * @author David B. Bracewell
 */
public class ConditionalMultinomial implements BivariateDensity, Copyable<ConditionalMultinomial> {
   private static final long serialVersionUID = 1L;
   private final double alpha;
   private final int N;
   private final int M;
   private final int[][] counts;
   private final int[] sums;

   /**
    * Instantiates a new Conditional multinomial.
    *
    * @param N     the number of discrete events being conditioned by, i.e. <code>X</code> in <code>P(E|X=x)</code>
    * @param M     the number of discrete events <code>E</code> in <code>P(E|X=x)</code>
    * @param alpha the smoothing parameter
    */
   public ConditionalMultinomial(int N, int M, double alpha) {
      this.N = N;
      this.M = M;
      this.alpha = alpha;
      this.sums = new int[N];
      this.counts = new int[N][M];
   }

   /**
    * Gets the smoothing parameter
    *
    * @return the smoothing parameter
    */
   public double getAlpha() {
      return alpha;
   }

   /**
    * Gets the number of discrete events being conditioned by, i.e. <code>X</code> in <code>P(E|X=x)</code>
    *
    * @return the number of discrete events being conditioned by, i.e. <code>X</code> in <code>P(E|X=x)</code>
    */
   public int getN() {
      return N;
   }

   /**
    * Gets the number of discrete events <code>E</code> in <code>P(E|X=x)</code>
    *
    * @return the number of discrete events <code>E</code> in <code>P(E|X=x)</code>
    */
   public int getM() {
      return M;
   }

   /**
    * Increments the count for the given n and m
    *
    * @param n the item conditioned by
    * @param m the discrete variable of interest
    */
   public void increment(int n, int m) {
      increment(n, m, 1);
   }

   /**
    * Increments the count for the given n and m
    *
    * @param n      the item conditioned by
    * @param m      the discrete variable of interest
    * @param amount the amount to increment the count by
    */
   public void increment(int n, int m, int amount) {
      this.counts[n][m] += amount;
      this.sums[n] += amount;
   }

   /**
    * Decrements the count for the given n and m
    *
    * @param n the item conditioned by
    * @param m the discrete variable of interest
    */
   public void decrement(int n, int m) {
      increment(n, m, -1);
   }

   /**
    * Decrements the count for the given n and m
    *
    * @param n      the item conditioned by
    * @param m      the discrete variable of interest
    * @param amount the amount to decrement the count by
    */
   public void decrement(int n, int m, int amount) {
      increment(n, m, -amount);
   }

   /**
    * Gets the counts of items when conditioned by n
    *
    * @param n the conditioned by variable
    * @return the counts as an int array
    */
   public int[] counts(int n) {
      return Arrays.copyOf(counts[n], M);
   }

   /**
    * Gets the count of m conditioned by n
    *
    * @param n the conditioned by variable
    * @param m the discrete variable of interest
    * @return the count of m conditioned by n
    */
   public int count(int n, int m) {
      return counts[n][m];
   }

   /**
    * Gets the sum of the counts of items when conditioned by n
    *
    * @param n the conditioned by variable
    * @return the sum of the counts of items when conditioned by n
    */
   public double sum(int n) {
      return sums[n];
   }

   /**
    * Gets the probability of <code>P(E=m|X=n)</code>
    *
    * @param n the conditioned by variable
    * @param m the discrete variable of interest
    * @return the probability of <code>P(E=m|X=n)</code>
    */
   @Override
   public double probability(int n, int m) {
      return (counts[n][m] + alpha) / (sums[n] + M * alpha);
   }

   /**
    * Gets the probabilities of variables in <code>E</code> when conditioned by <code>X=n</code>
    *
    * @param n the conditioned by variable
    * @return the probabilities of variables in <code>E</code> when conditioned by <code>X=n</code>
    */
   public double[] probabilities(int n) {
      double[] p = new double[M];
      for (int m = 0; m < M; m++) {
         p[m] = probability(n, m);
      }
      return p;
   }

   @Override
   public ConditionalMultinomial copy() {
      ConditionalMultinomial copy = new ConditionalMultinomial(N, M, alpha);
      System.arraycopy(counts, 0, copy.counts, 0, counts.length);
      System.arraycopy(sums, 0, copy.sums, 0, sums.length);
      return copy;
   }

}// END OF ConditionalMultinomial
