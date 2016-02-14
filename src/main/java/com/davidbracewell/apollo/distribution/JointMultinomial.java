package com.davidbracewell.apollo.distribution;

import java.io.Serializable;
import java.util.Arrays;

/**
 * The type Joint multinomial.
 *
 * @author David B. Bracewell
 */
public class JointMultinomial implements Serializable {
  private static final long serialVersionUID = 1L;
  private final double alpha;
  private final int N;
  private int[][] counts;
  private long sum;

  /**
   * Instantiates a new Joint multinomial.
   *
   * @param N     the n
   * @param alpha the alpha
   */
  public JointMultinomial(int N, double alpha) {
    this.N = N;
    this.alpha = alpha;
    this.counts = new int[N][N];
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
    this.sum += amount;
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
    return Arrays.copyOf(counts[n], N);
  }

  /**
   * Sum double.
   *
   * @return the double
   */
  public double sum() {
    return sum;
  }

  /**
   * Probability double.
   *
   * @param n the n
   * @param m the m
   * @return the double
   */
  public double probability(int n, int m) {
    return (counts[n][m] + alpha) / (sum + N * alpha);
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

}// END OF JointMultinomial
