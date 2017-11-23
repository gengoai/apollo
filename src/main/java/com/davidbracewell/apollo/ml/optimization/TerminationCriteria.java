package com.davidbracewell.apollo.ml.optimization;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * The type Termination criteria.
 *
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public final class TerminationCriteria implements Serializable {
   private final LinkedList<Double> history = new LinkedList<>();
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   @Setter
   private int historySize = 5;
   @Getter
   @Setter
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
            converged = Math.abs(sumLoss - itr.next()) <= tolerance;
         }
      }
      history.addFirst(sumLoss);
      return converged;
   }

   /**
    * Last loss double.
    *
    * @return the double
    */
   public double lastLoss() {
      return history.getFirst();
   }


}// END OF TerminationCriteria
