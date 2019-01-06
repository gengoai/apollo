package com.gengoai.apollo.optimization;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * The type Termination criteria.
 *
 * @author David B. Bracewell
 */
public final class TerminationCriteria implements Serializable {
   private final LinkedList<Double> history = new LinkedList<>();
   private int maxIterations = 100;
   private int historySize = 5;
   private double tolerance = 1e-6;

   /**
    * Create termination criteria.
    *
    * @return the termination criteria
    */
   public static TerminationCriteria create() {
      return new TerminationCriteria();
   }

   /**
    * Check boolean.
    *
    * @param sumLoss the sum loss
    * @return the boolean
    */
   public boolean check(double sumLoss) {
      boolean converged = false;
      if (!Double.isFinite(sumLoss)) {
         System.err.println("Non Finite loss, aborting");
         return true;
      }
      if (history.size() >= historySize) {
         converged = Math.abs(sumLoss - history.removeLast()) <= tolerance;
         Iterator<Double> itr = history.iterator();
         while (converged && itr.hasNext()) {
            double n = itr.next();
            converged = Math.abs(sumLoss - n) <= tolerance //loss in tolerance
                           || sumLoss > n; //or we got worse
         }
      }
      history.addFirst(sumLoss);
      return converged;
   }

   public int historySize() {
      return this.historySize;
   }

   public TerminationCriteria historySize(int historySize) {
      this.historySize = historySize;
      return this;
   }

   /**
    * Last loss double.
    *
    * @return the double
    */
   public double lastLoss() {
      return history.getFirst();
   }


   public int maxIterations() {
      return this.maxIterations;
   }

   public TerminationCriteria maxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
   }

   public double tolerance() {
      return this.tolerance;
   }

   public TerminationCriteria tolerance(double tolerance) {
      this.tolerance = tolerance;
      return this;
   }
}// END OF TerminationCriteria
