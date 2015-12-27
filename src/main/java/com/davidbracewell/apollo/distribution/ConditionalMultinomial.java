package com.davidbracewell.apollo.distribution;

import java.io.Serializable;
import java.util.Arrays;

/**
 * The type Conditional multinomial.
 *
 * @author David B. Bracewell
 */
public class ConditionalMultinomial implements Serializable {
  private static final long serialVersionUID = 1L;
  private final double alpha;
  private final int N;
  private final int M;
  private int[][] counts;
  private int[] sums;

  /**
   * Instantiates a new Conditional multinomial.
   *
   * @param N     the n
   * @param M     the m
   * @param alpha the alpha
   */
  public ConditionalMultinomial(int N, int M, double alpha) {
    this.N = N;
    this.M = M;
    this.alpha = alpha;
    this.sums = new int[N];
    this.counts = new int[N][M];
  }

  /**
   * Increment.
   *
   * @param n the n
   * @param m the m
   */
  public void increment(int n, int m) {
    increment(n, m, 1);
  }

  /**
   * Increment.
   *
   * @param n      the n
   * @param m      the m
   * @param amount the amount
   */
  public void increment(int n, int m, int amount) {
    this.counts[n][m] += amount;
    this.sums[n] += amount;
  }

  /**
   * Decrement.
   *
   * @param n the n
   * @param m the m
   */
  public void decrement(int n, int m) {
    increment(n, m, -1);
  }

  /**
   * Decrement.
   *
   * @param n      the n
   * @param m      the m
   * @param amount the amount
   */
  public void decrement(int n, int m, int amount) {
    increment(n, m, -amount);
  }

  /**
   * Counts int [ ].
   *
   * @param n the n
   * @return the int [ ]
   */
  public int[] counts(int n) {
    return Arrays.copyOf(counts[n], M);
  }

  /**
   * Sum double.
   *
   * @param n the n
   * @return the double
   */
  public double sum(int n) {
    return sums[n];
  }

  /**
   * Probability double.
   *
   * @param n the n
   * @param m the m
   * @return the double
   */
  public double probability(int n, int m) {
    return (counts[n][m] + alpha) / (sums[n] + M * alpha);
  }

  /**
   * Log probability double.
   *
   * @param n the n
   * @param m the m
   * @return the double
   */
  public double logProbability(int n, int m) {
    return Math.log(probability(n, m));
  }

  /**
   * Unnormalized probability double.
   *
   * @param n the n
   * @param m the m
   * @return the double
   */
  public double unnormalizedProbability(int n, int m) {
    return counts[n][m] + alpha;
  }

  /**
   * Unnormalized log probability double.
   *
   * @param n the n
   * @param m the m
   * @return the double
   */
  public double unnormalizedLogProbability(int n, int m) {
    return Math.log(unnormalizedProbability(n, m));
  }

}// END OF ConditionalMultinomial
